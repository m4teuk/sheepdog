"""Unsupervised exploration on Sheepdog: PPO trained only on an intrinsic bonus (no task reward).

Bonuses (--bonus):
  icm  Curiosity (forward-model error). A model predicts the change in dog/sheep positions from
       (dog/sheep positions, sheep velocity, action); its squared error is the reward. Simplest
       ICM variant: raw state instead of learned features, no inverse model (Burda et al. 2018,
       "Large-Scale Study of Curiosity-Driven Learning").
  rnd  Novelty (Burda et al. 2018, RND). A predictor learns to match a fixed random network on
       dog/sheep positions; its error is the reward.
--random-policy skips training and logs the same diagnostics for uniform random actions (chance baseline).

Both bonuses are trained on the policy's own data, rescaled by a running std, and treated as
non-episodic (returns ignore episode ends, as in RND). Transitions that end in an auto-reset get no
bonus and are not used to train the bonus model: their "next state" is a random new start, and
rewarding that would reward ending episodes (i.e. leak the task reward via success resets).

Logged diagnostics (not used for training):
  success  - of all episodes finished since the last log line, the fraction that ended at the goal
  near     - fraction of steps with a dog within flee_radius of a sheep (is the dog seeking sheep?)
  sheep_mv / dog_mv - how far sheep / dog moved over one rollout segment (arena units)

Run on a GPU node:
    uv run --extra viz python scripts/intrinsic_ppo.py --bonus icm --env noise_std=0
    uv run --extra viz python scripts/intrinsic_ppo.py --random-policy --env noise_std=0
Outputs (metrics.csv, params.pkl, example episodes as .npz/.png/.gif) go to outputs/intrinsic_ppo/<tag>-<time>/.
"""

import argparse
import csv
import json
import pickle
import time
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sheepdog import SheepdogConfig
from sheepdog import jax_env as E

BONUSES = ("icm", "rnd")


@dataclass
class Args:
    bonus: str = "icm"  # one of BONUSES
    random_policy: bool = False
    seed: int = 0
    total_steps: int = 20_000_000
    num_envs: int = 1024
    num_steps: int = 128  # rollout length per update
    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 32
    hidden: int = 256
    # bonus model
    bonus_lr: float = 3e-4
    rnd_dim: int = 128
    # output
    log_every: int = 5
    n_eval_episodes: int = 4
    out: str = "outputs/intrinsic_ppo"


# --- tiny MLP library -------------------------------------------------------------------


def mlp_init(key, sizes, last_scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        params.append((jax.nn.initializers.orthogonal(scale)(k, (n_in, n_out)), jnp.zeros(n_out)))
    return params


def mlp(params, x, act=jnp.tanh):
    for w, b in params[:-1]:
        x = act(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def gauss_logp(a, mean, log_std):
    return (-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def update_stats(stats, x, w):
    """Running mean/var (parallel Welford) of the values in x with weight w (0/1 mask)."""
    mean, var, count = stats
    b_count = w.sum()
    b_mean = (w * x).sum() / jnp.maximum(b_count, 1)
    b_var = (w * (x - b_mean) ** 2).sum() / jnp.maximum(b_count, 1)
    delta, tot = b_mean - mean, count + b_count
    m2 = var * count + b_var * b_count + delta**2 * count * b_count / tot
    return mean + delta * b_count / tot, m2 / tot, tot


# --- intrinsic bonuses ------------------------------------------------------------------
# init(key) -> (trainable params, fixed params)
# error(trainable, fixed, obs, action, next_obs) -> per-transition error = bonus (before scaling)


def make_bonus(kind: str, cfg: SheepdogConfig, hidden: int, rnd_dim: int):
    D, S, L, h = cfg.n_dogs, cfg.n_sheep, cfg.arena_size, hidden
    pos_dim = 2 * (D + S)  # flat_obs layout: [dogs, sheep, sheep_vel, goal]
    state_dim = 2 * D + 4 * S  # positions + sheep velocities (no goal)
    act_dim = 2 * D
    relu = jax.nn.relu

    if kind == "rnd":

        def init(key):
            k1, k2 = jax.random.split(key)
            return mlp_init(k1, [pos_dim, h, h, rnd_dim]), mlp_init(k2, [pos_dim, h, h, rnd_dim])

        def error(train, fixed, obs, act, next_obs):
            x = next_obs[..., :pos_dim]
            return ((mlp(train, x, relu) - mlp(fixed, x, relu)) ** 2).mean(-1)

    elif kind == "icm":
        # Predict position deltas in units of "one full-speed step", so dog and sheep errors are comparable.
        max_step = jnp.concatenate(
            [
                jnp.full(2 * D, 2 * cfg.dog_max_speed * cfg.dt / L),
                jnp.full(2 * S, 2 * cfg.sheep_max_speed * cfg.dt / L),
            ]
        )

        def init(key):
            return mlp_init(key, [state_dim + act_dim, h, h, pos_dim]), {}

        def error(train, fixed, obs, act, next_obs):
            x = jnp.concatenate([obs[..., :state_dim], jnp.clip(act, -1.0, 1.0)], axis=-1)
            target = (next_obs[..., :pos_dim] - obs[..., :pos_dim]) / max_step
            return ((mlp(train, x, relu) - target) ** 2).mean(-1)

    else:
        raise ValueError(f"unknown bonus {kind!r}")
    return init, error


# --- training ---------------------------------------------------------------------------


def make_train(args: Args, cfg: SheepdogConfig):
    N, T = args.num_envs, args.num_steps
    D, S, L = cfg.n_dogs, cfg.n_sheep, cfg.arena_size
    obs_dim, act_dim = E.obs_size(cfg), 2 * D

    env_reset = jax.vmap(partial(E.reset, cfg))
    env_step = jax.vmap(partial(E.step_autoreset, cfg))
    env_obs = jax.vmap(partial(E.flat_obs, cfg))
    opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))
    bonus_opt = optax.adam(args.bonus_lr)
    bonus_init, bonus_error = make_bonus(args.bonus, cfg, args.hidden, args.rnd_dim)

    def policy(params, obs):
        return mlp(params["pi"], obs), params["log_std"], mlp(params["v"], obs)[..., 0]

    def init(key):
        k_pi, k_v, k_bonus, k_env, key = jax.random.split(key, 5)
        h = args.hidden
        params = {
            "pi": mlp_init(k_pi, [obs_dim, h, h, act_dim], last_scale=0.01),
            "v": mlp_init(k_v, [obs_dim, h, h, 1], last_scale=1.0),
            "log_std": jnp.zeros(act_dim),
        }
        bonus_train, bonus_fixed = bonus_init(k_bonus)
        env_state = env_reset(jax.random.split(k_env, N))
        return {
            "params": params,
            "opt_state": opt.init(params),
            "bonus_train": bonus_train,
            "bonus_fixed": bonus_fixed,
            "bonus_opt_state": bonus_opt.init(bonus_train),
            "env_state": env_state,
            "obs": env_obs(env_state),
            "rew_stats": (jnp.float32(0.0), jnp.float32(1.0), jnp.float32(1e-4)),
            "key": key,
        }

    def ppo_loss(params, mb):
        mean, log_std, v = policy(params, mb["obs"])
        ratio = jnp.exp(gauss_logp(mb["action"], mean, log_std) - mb["logp"])
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv).mean()
        vl = 0.5 * ((v - mb["ret"]) ** 2).mean()
        ent = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        return pg + args.vf_coef * vl - args.ent_coef * ent, (pg, vl, ent)

    def bonus_loss(train, fixed, mb):
        err = bonus_error(train, fixed, mb["obs"], mb["action"], mb["next_obs"])
        return (err * mb["valid"]).sum() / jnp.maximum(mb["valid"].sum(), 1)

    @jax.jit
    def update(c):
        # 1. Rollout.
        def rollout_step(carry, _):
            env_state, obs, key = carry
            key, k_a, k_e = jax.random.split(key, 3)
            mean, log_std, v = policy(c["params"], obs)
            if args.random_policy:
                a = jax.random.uniform(k_a, mean.shape, minval=-1.0, maxval=1.0)
            else:
                a = mean + jnp.exp(log_std) * jax.random.normal(k_a, mean.shape)
            env_state, next_obs, done, info = env_step(env_state, a.reshape(N, D, 2), jax.random.split(k_e, N))
            tr = {
                "obs": obs, "action": a, "logp": gauss_logp(a, mean, log_std), "value": v,
                "next_obs": next_obs, "done": done, "success": info["success"],
            }  # fmt: skip
            return (env_state, next_obs, key), tr

        (env_state, obs, key), tr = jax.lax.scan(rollout_step, (c["env_state"], c["obs"], c["key"]), None, T)

        # 2. Intrinsic reward, zero on auto-reset transitions.
        valid = (~tr["done"]).astype(jnp.float32)
        r_raw = bonus_error(c["bonus_train"], c["bonus_fixed"], tr["obs"], tr["action"], tr["next_obs"]) * valid
        rew_stats = update_stats(c["rew_stats"], r_raw, valid)
        r = r_raw / jnp.sqrt(rew_stats[1] + 1e-8)

        # 3. Non-episodic GAE (intrinsic returns ignore episode boundaries).
        _, _, last_v = policy(c["params"], obs)

        def gae_step(carry, x):
            adv, next_v = carry
            r_t, v_t = x
            adv = r_t + args.gamma * next_v - v_t + args.gamma * args.gae_lambda * adv
            return (adv, v_t), adv

        _, adv = jax.lax.scan(gae_step, (jnp.zeros(N), last_v), (r, tr["value"]), reverse=True)
        batch = {
            "obs": tr["obs"], "action": tr["action"], "logp": tr["logp"], "adv": adv,
            "ret": adv + tr["value"], "next_obs": tr["next_obs"], "valid": valid,
        }  # fmt: skip
        batch = jax.tree.map(lambda x: x.reshape(T * N, *x.shape[2:]), batch)

        # 4. PPO + bonus-model updates on shuffled minibatches.
        def epoch(carry, _):
            params, opt_state, b_train, b_opt_state, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, T * N)
            mbs = jax.tree.map(lambda x: x[perm].reshape(args.num_minibatches, -1, *x.shape[1:]), batch)

            def mb_step(carry, mb):
                params, opt_state, b_train, b_opt_state = carry
                (_, aux), g = jax.value_and_grad(ppo_loss, has_aux=True)(params, mb)
                upd, opt_state = opt.update(g, opt_state, params)
                params = optax.apply_updates(params, upd)
                b_loss, bg = jax.value_and_grad(bonus_loss)(b_train, c["bonus_fixed"], mb)
                bupd, b_opt_state = bonus_opt.update(bg, b_opt_state, b_train)
                b_train = optax.apply_updates(b_train, bupd)
                return (params, opt_state, b_train, b_opt_state), (*aux, b_loss)

            (params, opt_state, b_train, b_opt_state), stats = jax.lax.scan(
                mb_step, (params, opt_state, b_train, b_opt_state), mbs
            )
            return (params, opt_state, b_train, b_opt_state, key), stats

        if args.random_policy:
            params, opt_state, b_train, b_opt_state = c["params"], c["opt_state"], c["bonus_train"], c["bonus_opt_state"]
            pg = vl = ent = b_loss = jnp.zeros(1)
        else:
            (params, opt_state, b_train, b_opt_state, key), (pg, vl, ent, b_loss) = jax.lax.scan(
                epoch,
                (c["params"], c["opt_state"], c["bonus_train"], c["bonus_opt_state"], key),
                None,
                args.update_epochs,
            )

        # 5. Diagnostics (not used for training).
        n_done = tr["done"].sum()
        pos = (tr["next_obs"] + 1.0) * (L / 2)  # back to arena units
        dog_xy = pos[..., : 2 * D].reshape(T, N, D, 2)
        sheep_xy = pos[..., 2 * D : 2 * (D + S)].reshape(T, N, S, 2)
        no_reset = ~tr["done"].any(0)  # envs without an episode boundary in this segment
        moved = lambda xy: (jnp.linalg.norm(xy[-1] - xy[0], axis=-1).mean(-1) * no_reset).sum() / jnp.maximum(no_reset.sum(), 1)  # noqa: E731
        dog_sheep_dist = jnp.linalg.norm(dog_xy[:, :, :, None, :] - sheep_xy[:, :, None, :, :], axis=-1)  # (T,N,D,S)
        metrics = {
            "r_int": r_raw.sum() / jnp.maximum(valid.sum(), 1),
            "bonus_loss": b_loss.mean(),
            "pg_loss": pg.mean(),
            "v_loss": vl.mean(),
            "act_std": jnp.exp(params["log_std"]).mean(),
            "episodes": n_done,
            "successes": (tr["done"] & tr["success"]).sum(),
            "near": (dog_sheep_dist < cfg.flee_radius).any(axis=(2, 3)).mean(),
            "dog_mv": moved(dog_xy),
            "sheep_mv": moved(sheep_xy),
        }
        new_c = {
            **c, "params": params, "opt_state": opt_state, "bonus_train": b_train, "bonus_opt_state": b_opt_state,
            "env_state": env_state, "obs": obs, "rew_stats": rew_stats, "key": key,
        }  # fmt: skip
        return new_c, metrics

    return init, update, policy


def record_episodes(cfg, policy, params, key, n, random_policy=False):
    """Roll out the (stochastic) policy without auto-reset; return viz-compatible dicts."""

    @jax.jit
    def episode(key):
        k0, key = jax.random.split(key)
        s0 = E.reset(cfg, k0)

        def f(s, k):
            k_a, k_s = jax.random.split(k)
            mean, log_std, _ = policy(params, E.flat_obs(cfg, s))
            if random_policy:
                a = jax.random.uniform(k_a, mean.shape, minval=-1.0, maxval=1.0)
            else:
                a = mean + jnp.exp(log_std) * jax.random.normal(k_a, mean.shape)
            s2 = E.step(cfg, s, a, k_s)
            return s2, (s2, E.is_success(cfg, s2))

        _, (states, success) = jax.lax.scan(f, s0, jax.random.split(key, cfg.max_episode_steps))
        states = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), s0, states)
        return states, success

    trajs = []
    for k in jax.random.split(key, n):
        states, success = jax.device_get(episode(k))
        success = np.asarray(success)
        T = int(np.argmax(success)) + 1 if success.any() else cfg.max_episode_steps
        trajs.append(
            {
                "dogs": states.dogs[: T + 1], "sheep": states.sheep[: T + 1],
                "sheep_vel": states.sheep_vel[: T + 1], "goal": states.goal[: T + 1],
                "success": np.bool_(success.any()), "length": np.int64(T),
            }  # fmt: skip
        )
    return trajs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Args):
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):
            p.add_argument(flag, action="store_true", default=f.default)
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)
    p.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE",
                   help="SheepdogConfig overrides, e.g. --env noise_std=0 n_sheep=3")  # fmt: skip
    ns = vars(p.parse_args())
    env_overrides = ns.pop("env")
    args = Args(**ns)
    if args.bonus not in BONUSES:
        p.error(f"--bonus must be one of {BONUSES}")
    cfg_types = {f.name: f.type for f in fields(SheepdogConfig)}
    overrides = {}
    for kv in env_overrides:
        k, v = kv.split("=", 1)
        if k not in cfg_types:
            p.error(f"unknown env parameter {k!r}; choose from {sorted(cfg_types)}")
        overrides[k] = (int if cfg_types[k] in (int, "int") else float)(v)
    return args, SheepdogConfig(**overrides)


def main():
    args, cfg = parse_args()
    tag = "random" if args.random_policy else args.bonus
    run_dir = Path(args.out) / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps({"args": asdict(args), "env": asdict(cfg)}, indent=2))
    print(f"devices: {jax.devices()}\nrun dir: {run_dir}")

    init, update, policy = make_train(args, cfg)
    c = init(jax.random.key(args.seed))
    n_updates = args.total_steps // (args.num_envs * args.num_steps)

    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        writer = None
        t0 = time.time()
        win_eps = win_succ = 0  # episode counts since the last log line
        for i in range(1, n_updates + 1):
            c, m = update(c)
            m = {k: float(v) for k, v in jax.device_get(m).items()}
            win_eps += m.pop("episodes")
            win_succ += m.pop("successes")
            if i % args.log_every == 0 or i == 1 or i == n_updates:
                m["episodes"] = win_eps
                m["success"] = win_succ / max(win_eps, 1)
                win_eps = win_succ = 0
                steps = i * args.num_envs * args.num_steps
                row = {"update": i, "steps": steps, "sps": int(steps / (time.time() - t0)), **m}
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
                print(
                    f"[{i:4d}/{n_updates}] {steps / 1e6:5.1f}M  sps {row['sps']:>8,}  "
                    f"r_int {m['r_int']:.4f}  bonus {m['bonus_loss']:.4f}  v {m['v_loss']:.3f}  "
                    f"std {m['act_std']:.2f}  near {m['near']:.3f}  dog_mv {m['dog_mv']:.2f}  "
                    f"sheep_mv {m['sheep_mv']:.2f}  eps {int(m['episodes']):5d}  success {m['success']:.3f}"
                )

    params = jax.device_get(c["params"])
    with open(run_dir / "params.pkl", "wb") as fh:
        pickle.dump(params, fh)

    trajs = record_episodes(
        cfg, policy, params, jax.random.key(args.seed + 1), args.n_eval_episodes, args.random_policy
    )
    try:
        from sheepdog.viz import animate, plot_trajectory
    except ImportError:
        animate = plot_trajectory = None
        print("matplotlib not installed (use --extra viz); saving episodes as .npz only")
    for j, traj in enumerate(trajs):
        np.savez(run_dir / f"episode{j}.npz", **traj)
        if animate is not None:
            plot_trajectory(traj, cfg, run_dir / f"episode{j}.png")
            animate(traj, cfg, run_dir / f"episode{j}.gif", stride=2)
        print(f"episode {j}: {int(traj['length'])} steps, success={bool(traj['success'])}")
    print(f"done, outputs in {run_dir}")


if __name__ == "__main__":
    main()
