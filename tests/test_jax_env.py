from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sheepdog import SheepdogConfig, SheepdogEnv
from sheepdog import jax_env as J
from sheepdog.dynamics import step_state


@pytest.mark.parametrize(
    "overrides",
    [
        dict(n_dogs=1, n_sheep=1),
        # with 2 sheep each sheep always follows the other, so flocking is deterministic too
        dict(n_dogs=2, n_sheep=2, flocking_strength=1.0, follow_switch_steps=7),
    ],
)
def test_matches_numpy_without_noise(overrides):
    """Same start state + same actions -> same trajectory as the numpy env (noise off)."""
    cfg = SheepdogConfig(noise_std=0.0, sheep_inertia=0.5, **overrides)
    env = SheepdogEnv(cfg)
    env.reset(seed=3)
    np_state = env.get_state()
    j_state = J.from_numpy(np_state)
    jstep = jax.jit(partial(J.step, cfg))
    rng = np.random.default_rng(0)
    key = jax.random.key(0)
    for _ in range(200):
        a = rng.uniform(-1, 1, size=(cfg.n_dogs, 2))
        np_state = step_state(cfg, np_state, a, rng)
        key, sub = jax.random.split(key)
        j_state = jstep(j_state, jnp.asarray(a, jnp.float32), sub)
        got = J.to_numpy(j_state)
        np.testing.assert_allclose(got.dogs, np_state.dogs, atol=1e-3)
        np.testing.assert_allclose(got.sheep, np_state.sheep, atol=1e-3)
        np.testing.assert_array_equal(got.follow_target, np_state.follow_target)
        assert got.t == np_state.t


def test_reset_batched_respects_spawn_distance():
    cfg = SheepdogConfig(n_sheep=5)
    states = jax.jit(jax.vmap(partial(J.reset, cfg)))(jax.random.split(jax.random.key(0), 512))
    d = jnp.linalg.norm(states.sheep - states.goal[:, None, :], axis=-1)
    assert bool((d >= cfg.min_start_sheep_goal_dist).all())
    assert states.dogs.shape == (512, 1, 2)
    assert bool(((states.sheep >= 0) & (states.sheep <= cfg.arena_size)).all())


def test_flat_obs_shape_and_range():
    cfg = SheepdogConfig(n_dogs=2, n_sheep=3)
    s = J.reset(cfg, jax.random.key(1))
    o = J.flat_obs(cfg, s)
    assert o.shape == (J.obs_size(cfg),)
    assert bool((jnp.abs(o) <= 1.0 + 1e-6).all())


def test_autoreset_on_success():
    cfg = SheepdogConfig(noise_std=0.0, sheep_inertia=0.0)
    s = J.reset(cfg, jax.random.key(0))
    s = s._replace(sheep=s.goal[None, :] + 0.1, dogs=jnp.zeros((1, 2)), t=jnp.int32(17))
    new, obs, done, info = J.step_autoreset(cfg, s, jnp.zeros((1, 2)), jax.random.key(1))
    assert bool(done) and bool(info["success"]) and not bool(info["truncated"])
    assert int(new.t) == 0
    np.testing.assert_allclose(obs, J.flat_obs(cfg, new))


def test_autoreset_on_time_limit():
    cfg = SheepdogConfig(max_episode_steps=5)
    step = jax.jit(partial(J.step_autoreset, cfg))
    s = J.reset(cfg, jax.random.key(0))
    s = s._replace(dogs=jnp.zeros((1, 2)), sheep=jnp.full((1, 2), 9.0), goal=jnp.array([1.0, 9.0]))
    for i in range(5):
        s, _, done, info = step(s, jnp.zeros((1, 2)), jax.random.key(i))
        assert bool(done) == (i == 4)
    assert bool(info["truncated"]) and int(s.t) == 0


def test_vmapped_rollout_runs():
    cfg = SheepdogConfig(n_dogs=2, n_sheep=4, flocking_strength=0.5)
    n = 64
    step = jax.jit(jax.vmap(partial(J.step_autoreset, cfg)))
    states = jax.vmap(partial(J.reset, cfg))(jax.random.split(jax.random.key(0), n))
    key = jax.random.key(1)
    for _ in range(50):
        key, ka, ks = jax.random.split(key, 3)
        a = jax.random.uniform(ka, (n, cfg.n_dogs, 2), minval=-1, maxval=1)
        states, obs, done, _ = step(states, a, jax.random.split(ks, n))
    assert obs.shape == (n, J.obs_size(cfg))
    assert bool(jnp.isfinite(obs).all())
