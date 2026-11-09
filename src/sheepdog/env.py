from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SheepdogConfig
from .dynamics import State, is_success, sample_initial_state, sheep_goal_dist, step_state


class SheepdogEnv(gym.Env):
    """Dogs herd sheep to a goal point in a 2D continuous arena. Full state is observed.

    Observation (dict, float32):
        dogs       (n_dogs, 2)   positions
        sheep      (n_sheep, 2)  positions
        sheep_vel  (n_sheep, 2)  velocities over the last step
        goal       (2,)          destination
    Action: (n_dogs, 2) in [-1, 1], velocity command for each dog (scaled by dog_max_speed).
    Reward: 1.0 on the step all sheep are within goal_radius of the goal, else 0.0.
        Placeholder only; training is expected to use its own (unsupervised / goal-conditioned)
        reward. info["success"] carries the same signal.
    Termination on success; truncation after max_episode_steps.

    reset(options=...) can pin any of "dogs", "sheep", "goal" to given positions.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SheepdogConfig | None = None, **overrides):
        self.cfg = replace(config or SheepdogConfig(), **overrides)
        c = self.cfg
        L = np.float32(c.arena_size)
        v = np.float32(c.sheep_max_speed)
        self.observation_space = spaces.Dict(
            {
                "dogs": spaces.Box(0.0, L, (c.n_dogs, 2), np.float32),
                "sheep": spaces.Box(0.0, L, (c.n_sheep, 2), np.float32),
                "sheep_vel": spaces.Box(-v, v, (c.n_sheep, 2), np.float32),
                "goal": spaces.Box(0.0, L, (2,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (c.n_dogs, 2), np.float32)
        self._state: State | None = None

    # --- Gymnasium API -------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = sample_initial_state(self.cfg, self.np_random)
        if options:
            state = self._apply_reset_options(state, options)
        self._state = state
        return self._obs(), self._info()

    def step(self, action):
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        self._state = step_state(self.cfg, self._state, action, self.np_random)
        success = is_success(self.cfg, self._state)
        terminated = success
        truncated = (not terminated) and self._state.t >= self.cfg.max_episode_steps
        return self._obs(), float(success), terminated, truncated, self._info()

    # --- State access ----------------------------------------------------------------

    def get_state(self) -> State:
        return self._state.copy()

    def set_state(self, state: State) -> None:
        self._state = state.copy()

    # --- Helpers ---------------------------------------------------------------------

    def _apply_reset_options(self, state: State, options: dict) -> State:
        unknown = set(options) - {"dogs", "sheep", "goal"}
        if unknown:
            raise ValueError(f"unknown reset options: {sorted(unknown)}")
        shapes = {"dogs": (self.cfg.n_dogs, 2), "sheep": (self.cfg.n_sheep, 2), "goal": (2,)}
        fixed = {}
        for key, value in options.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != shapes[key]:
                raise ValueError(f"options[{key!r}] has shape {arr.shape}, expected {shapes[key]}")
            fixed[key] = np.clip(arr, 0.0, self.cfg.arena_size)
        return replace(state, **fixed)

    def _obs(self) -> dict:
        s = self._state
        return {
            "dogs": s.dogs.astype(np.float32),
            "sheep": s.sheep.astype(np.float32),
            "sheep_vel": s.sheep_vel.astype(np.float32),
            "goal": s.goal.astype(np.float32),
        }

    def _info(self) -> dict:
        return {
            "success": is_success(self.cfg, self._state),
            "sheep_goal_dist": sheep_goal_dist(self._state),
            "goal": self._state.goal.copy(),
        }
