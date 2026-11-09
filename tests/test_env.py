import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import sheepdog  # noqa: F401  (registers Sheepdog-v0)
from sheepdog import SheepdogConfig, SheepdogEnv

NO_NOISE = dict(noise_std=0.0, sheep_inertia=0.0)


def test_gymnasium_env_checker():
    check_env(SheepdogEnv(), skip_render_check=True)


def test_make_by_id():
    env = gym.make("Sheepdog-v0", n_sheep=3)
    obs, _ = env.reset(seed=0)
    assert obs["sheep"].shape == (3, 2)


@pytest.mark.parametrize("n_dogs,n_sheep", [(1, 1), (2, 5)])
def test_shapes_and_bounds(n_dogs, n_sheep):
    env = SheepdogEnv(n_dogs=n_dogs, n_sheep=n_sheep)
    obs, info = env.reset(seed=0)
    assert obs in env.observation_space
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs in env.observation_space
        if terminated or truncated:
            break


def test_initial_sheep_away_from_goal():
    env = SheepdogEnv(n_sheep=10)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        assert (info["sheep_goal_dist"] >= env.cfg.min_start_sheep_goal_dist).all()
        assert not info["success"]


def test_seeding_is_deterministic():
    def rollout(seed):
        env = SheepdogEnv(n_sheep=4, flocking_strength=0.5)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return [env.step(env.action_space.sample())[0]["sheep"] for _ in range(50)]

    np.testing.assert_array_equal(rollout(1), rollout(1))
    assert not np.array_equal(rollout(1)[-1], rollout(2)[-1])


def test_sheep_flees_from_dog():
    env = SheepdogEnv(**NO_NOISE)
    env.reset(seed=0, options=dict(dogs=[[5.0, 5.0]], sheep=[[6.0, 5.0]], goal=[1.0, 1.0]))
    obs, *_ = env.step(np.zeros((1, 2)))
    assert obs["sheep"][0, 0] > 6.0  # pushed in +x, away from the dog
    assert abs(obs["sheep"][0, 1] - 5.0) < 1e-6


def test_sheep_ignores_far_dog():
    env = SheepdogEnv(**NO_NOISE)
    env.reset(seed=0, options=dict(dogs=[[0.0, 0.0]], sheep=[[8.0, 8.0]], goal=[1.0, 9.0]))
    obs, *_ = env.step(np.zeros((1, 2)))
    np.testing.assert_allclose(obs["sheep"], [[8.0, 8.0]])


def test_sheep_speed_capped():
    env = SheepdogEnv(sheep_inertia=0.0, flee_strength=100.0, noise_std=0.0)
    env.reset(seed=0, options=dict(dogs=[[5.0, 5.0]], sheep=[[5.1, 5.0]], goal=[1.0, 1.0]))
    obs, *_ = env.step(np.zeros((1, 2)))
    assert np.linalg.norm(obs["sheep_vel"][0]) <= env.cfg.sheep_max_speed + 1e-5


def test_dog_moves_with_action_and_stays_in_arena():
    env = SheepdogEnv()
    env.reset(seed=0, options=dict(dogs=[[5.0, 5.0]]))
    obs, *_ = env.step(np.array([[1.0, -1.0]]))
    step = env.cfg.dog_max_speed * env.cfg.dt
    np.testing.assert_allclose(obs["dogs"][0], [5.0 + step, 5.0 - step], atol=1e-5)
    for _ in range(100):
        obs, *_ = env.step(np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(obs["dogs"][0], [env.cfg.arena_size] * 2)


def test_success_terminates():
    env = SheepdogEnv(**NO_NOISE)
    env.reset(seed=0, options=dict(dogs=[[0.0, 0.0]], sheep=[[5.0, 5.0]], goal=[5.2, 5.0]))
    _, reward, terminated, truncated, info = env.step(np.zeros((1, 2)))
    assert terminated and not truncated and info["success"] and reward == 1.0


def test_truncation():
    env = SheepdogEnv(max_episode_steps=10, **NO_NOISE)
    env.reset(seed=0, options=dict(dogs=[[0.0, 0.0]], sheep=[[9.0, 9.0]], goal=[1.0, 9.0]))
    for i in range(10):
        _, _, terminated, truncated, _ = env.step(np.zeros((1, 2)))
        assert not terminated
        assert truncated == (i == 9)


def test_flocking_off_by_default():
    assert SheepdogConfig().flocking_strength == 0.0


def test_flocking_pulls_sheep_together():
    opts = dict(dogs=[[0.0, 0.0]], sheep=[[3.0, 5.0], [7.0, 5.0]], goal=[5.0, 9.0])
    gaps = {}
    for strength in (0.0, 1.0):
        env = SheepdogEnv(n_sheep=2, flocking_strength=strength, **NO_NOISE)
        env.reset(seed=0, options=opts)
        for _ in range(10):
            obs, *_ = env.step(np.zeros((1, 2)))
        gaps[strength] = np.linalg.norm(obs["sheep"][0] - obs["sheep"][1])
    assert gaps[0.0] == pytest.approx(4.0)
    assert gaps[1.0] < 4.0


def test_get_set_state_roundtrip():
    env = SheepdogEnv(n_sheep=3, flocking_strength=0.5)
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    saved = env.get_state()
    rng_state = env.np_random.bit_generator.state
    a = [env.step(np.zeros((1, 2)))[0]["sheep"] for _ in range(5)]
    env.set_state(saved)
    env.np_random.bit_generator.state = rng_state
    b = [env.step(np.zeros((1, 2)))[0]["sheep"] for _ in range(5)]
    np.testing.assert_array_equal(a, b)


def test_bad_reset_options():
    env = SheepdogEnv()
    with pytest.raises(ValueError):
        env.reset(options=dict(sheep=[[1.0, 1.0], [2.0, 2.0]]))
    with pytest.raises(ValueError):
        env.reset(options=dict(cat=[1.0, 1.0]))
