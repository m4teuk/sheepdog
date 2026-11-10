"""JAX port of dynamics.py: same model, pure functions, jit/vmap-friendly.

The config is a static Python object (bind it with functools.partial before jit/vmap):

    step = jax.jit(jax.vmap(partial(step_autoreset, cfg)))
    states = jax.vmap(partial(reset, cfg))(jax.random.split(key, n_envs))

Random streams differ from the numpy env, so trajectories match only when noise is off
(see tests/test_jax_env.py).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import dynamics
from .config import SheepdogConfig

_EPS = 1e-8


class JState(NamedTuple):
    dogs: jax.Array  # (n_dogs, 2)
    sheep: jax.Array  # (n_sheep, 2)
    sheep_vel: jax.Array  # (n_sheep, 2)
    goal: jax.Array  # (2,)
    follow_target: jax.Array  # (n_sheep,) int32, -1 if none
    follow_timer: jax.Array  # (n_sheep,) int32
    t: jax.Array  # () int32


def _unit(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, _EPS), norm[..., 0]


def sample_follow_targets(n_sheep: int, key) -> jax.Array:
    if n_sheep == 1:
        return jnp.full(1, -1, dtype=jnp.int32)
    idx = jax.random.randint(key, (n_sheep,), 0, n_sheep - 1)
    return idx + (idx >= jnp.arange(n_sheep))  # skip self


def reset(cfg: SheepdogConfig, key) -> JState:
    L = cfg.arena_size
    k_goal, k_dogs, k_sheep, k_loop, k_follow, k_timer = jax.random.split(key, 6)
    goal = jax.random.uniform(k_goal, (2,), minval=0.0, maxval=L)
    dogs = jax.random.uniform(k_dogs, (cfg.n_dogs, 2), minval=0.0, maxval=L)
    sheep = jax.random.uniform(k_sheep, (cfg.n_sheep, 2), minval=0.0, maxval=L)

    def too_close(s):
        return jnp.linalg.norm(s - goal, axis=-1) < cfg.min_start_sheep_goal_dist

    def resample(carry):  # rejection-sample sheep that start too close to the goal
        s, k = carry
        k, sub = jax.random.split(k)
        fresh = jax.random.uniform(sub, s.shape, minval=0.0, maxval=L)
        return jnp.where(too_close(s)[:, None], fresh, s), k

    sheep, _ = jax.lax.while_loop(lambda c: too_close(c[0]).any(), resample, (sheep, k_loop))
    return JState(
        dogs=dogs,
        sheep=sheep,
        sheep_vel=jnp.zeros((cfg.n_sheep, 2)),
        goal=goal,
        follow_target=sample_follow_targets(cfg.n_sheep, k_follow).astype(jnp.int32),
        follow_timer=jax.random.randint(k_timer, (cfg.n_sheep,), 1, cfg.follow_switch_steps + 1),
        t=jnp.int32(0),
    )


def flee_velocity(cfg: SheepdogConfig, sheep, dogs):
    away, dist = _unit(sheep[:, None, :] - dogs[None, :, :])  # (S, D, 2), (S, D)
    mag = cfg.flee_strength * jnp.clip(1.0 - dist / cfg.flee_radius, 0.0, 1.0)
    return (mag[..., None] * away).sum(axis=1)


def flock_velocity(cfg: SheepdogConfig, sheep, follow_target):
    if cfg.flocking_strength == 0.0 or cfg.n_sheep == 1:  # static: decided at trace time
        return jnp.zeros_like(sheep)
    toward, _ = _unit(sheep[follow_target] - sheep)
    return cfg.flocking_strength * toward


def step(cfg: SheepdogConfig, state: JState, action, key) -> JState:
    """One step, no auto-reset. action: (n_dogs, 2) (or flat) in [-1, 1]."""
    L, dt = cfg.arena_size, cfg.dt
    k_follow, k_noise = jax.random.split(key)

    action = jnp.clip(jnp.reshape(action, (cfg.n_dogs, 2)), -1.0, 1.0)
    dogs = jnp.clip(state.dogs + action * cfg.dog_max_speed * dt, 0.0, L)

    timer = state.follow_timer - 1
    switch = timer <= 0
    follow_target = jnp.where(switch, sample_follow_targets(cfg.n_sheep, k_follow), state.follow_target)
    timer = jnp.where(switch, cfg.follow_switch_steps, timer)

    desired = (
        flee_velocity(cfg, state.sheep, dogs)
        + flock_velocity(cfg, state.sheep, follow_target)
        + cfg.noise_std * jax.random.normal(k_noise, state.sheep.shape)
    )
    vel = cfg.sheep_inertia * state.sheep_vel + (1.0 - cfg.sheep_inertia) * desired
    direction, speed = _unit(vel)
    vel = direction * jnp.minimum(speed, cfg.sheep_max_speed)[:, None]
    sheep = jnp.clip(state.sheep + vel * dt, 0.0, L)

    return JState(
        dogs=dogs,
        sheep=sheep,
        sheep_vel=(sheep - state.sheep) / dt,
        goal=state.goal,
        follow_target=follow_target.astype(jnp.int32),
        follow_timer=timer.astype(jnp.int32),
        t=state.t + 1,
    )


def is_success(cfg: SheepdogConfig, state: JState):
    return (jnp.linalg.norm(state.sheep - state.goal, axis=-1) <= cfg.goal_radius).all()


def obs_size(cfg: SheepdogConfig) -> int:
    return 2 * cfg.n_dogs + 4 * cfg.n_sheep + 2


def flat_obs(cfg: SheepdogConfig, state: JState):
    """Flat observation scaled to about [-1, 1]:
    [dogs (2*n_dogs), sheep (2*n_sheep), sheep_vel (2*n_sheep), goal (2)].
    Positions map [0, L] -> [-1, 1]; velocities are divided by sheep_max_speed."""
    pos = lambda p: (p / cfg.arena_size) * 2.0 - 1.0  # noqa: E731
    return jnp.concatenate(
        [
            pos(state.dogs).ravel(),
            pos(state.sheep).ravel(),
            (state.sheep_vel / cfg.sheep_max_speed).ravel(),
            pos(state.goal),
        ]
    )


def step_autoreset(cfg: SheepdogConfig, state: JState, action, key):
    """Step; if the episode ended (success or time limit), return a freshly reset state instead.

    Returns (state, flat_obs, done, info). info["success"] / ["truncated"] describe the step that
    just happened (before any reset)."""
    k_step, k_reset = jax.random.split(key)
    nxt = step(cfg, state, action, k_step)
    success = is_success(cfg, nxt)
    truncated = (~success) & (nxt.t >= cfg.max_episode_steps)
    done = success | truncated
    fresh = reset(cfg, k_reset)
    new = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, nxt)
    return new, flat_obs(cfg, new), done, {"success": success, "truncated": truncated}


def from_numpy(state: dynamics.State) -> JState:
    return JState(
        dogs=jnp.asarray(state.dogs, jnp.float32),
        sheep=jnp.asarray(state.sheep, jnp.float32),
        sheep_vel=jnp.asarray(state.sheep_vel, jnp.float32),
        goal=jnp.asarray(state.goal, jnp.float32),
        follow_target=jnp.asarray(state.follow_target, jnp.int32),
        follow_timer=jnp.asarray(state.follow_timer, jnp.int32),
        t=jnp.int32(state.t),
    )


def to_numpy(state: JState) -> dynamics.State:
    s = jax.device_get(state)
    return dynamics.State(
        dogs=np.asarray(s.dogs, np.float64),
        sheep=np.asarray(s.sheep, np.float64),
        sheep_vel=np.asarray(s.sheep_vel, np.float64),
        goal=np.asarray(s.goal, np.float64),
        follow_target=np.asarray(s.follow_target, np.int64),
        follow_timer=np.asarray(s.follow_timer, np.int64),
        t=int(s.t),
    )
