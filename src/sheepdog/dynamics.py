"""Pure simulation core: (config, state, action, rng) -> new state.

Kept free of Gymnasium and of in-place mutation so it can be ported to JAX
(np.random.Generator -> PRNG key, np -> jnp) and batched on the GPU later.
"""

from dataclasses import dataclass, replace

import numpy as np

from .config import SheepdogConfig

_EPS = 1e-8


@dataclass(frozen=True)
class State:
    dogs: np.ndarray  # (n_dogs, 2) positions
    sheep: np.ndarray  # (n_sheep, 2) positions
    sheep_vel: np.ndarray  # (n_sheep, 2) velocity actually moved last step
    goal: np.ndarray  # (2,) destination
    follow_target: np.ndarray  # (n_sheep,) int, index of followed sheep, -1 if none
    follow_timer: np.ndarray  # (n_sheep,) int, steps until a new target is picked
    t: int  # steps taken this episode

    def copy(self) -> "State":
        return State(
            dogs=self.dogs.copy(),
            sheep=self.sheep.copy(),
            sheep_vel=self.sheep_vel.copy(),
            goal=self.goal.copy(),
            follow_target=self.follow_target.copy(),
            follow_timer=self.follow_timer.copy(),
            t=self.t,
        )


def _unit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along the last axis and their norms (zero vectors stay zero)."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, _EPS), norm[..., 0]


def sample_follow_targets(n_sheep: int, rng: np.random.Generator) -> np.ndarray:
    """For each sheep, a uniformly random *other* sheep (-1 when there is only one)."""
    if n_sheep == 1:
        return np.full(1, -1, dtype=np.int64)
    idx = rng.integers(0, n_sheep - 1, size=n_sheep)
    return idx + (idx >= np.arange(n_sheep))  # skip self


def sample_initial_state(cfg: SheepdogConfig, rng: np.random.Generator) -> State:
    """Everything uniform in the arena; sheep at least min_start_sheep_goal_dist from the goal."""
    L = cfg.arena_size
    goal = rng.uniform(0.0, L, size=2)
    dogs = rng.uniform(0.0, L, size=(cfg.n_dogs, 2))
    sheep = rng.uniform(0.0, L, size=(cfg.n_sheep, 2))
    while True:  # rejection-sample sheep that start too close to the goal
        too_close = np.linalg.norm(sheep - goal, axis=-1) < cfg.min_start_sheep_goal_dist
        if not too_close.any():
            break
        sheep[too_close] = rng.uniform(0.0, L, size=(int(too_close.sum()), 2))
    return State(
        dogs=dogs,
        sheep=sheep,
        sheep_vel=np.zeros((cfg.n_sheep, 2)),
        goal=goal,
        follow_target=sample_follow_targets(cfg.n_sheep, rng),
        # random phase so sheep don't all switch targets on the same step
        follow_timer=rng.integers(1, cfg.follow_switch_steps + 1, size=cfg.n_sheep),
        t=0,
    )


def flee_velocity(cfg: SheepdogConfig, sheep: np.ndarray, dogs: np.ndarray) -> np.ndarray:
    """Push away from every dog within flee_radius, strength falling off linearly to 0 at the radius."""
    away, dist = _unit(sheep[:, None, :] - dogs[None, :, :])  # (S, D, 2), (S, D)
    mag = cfg.flee_strength * np.clip(1.0 - dist / cfg.flee_radius, 0.0, 1.0)
    return (mag[..., None] * away).sum(axis=1)


def flock_velocity(cfg: SheepdogConfig, sheep: np.ndarray, follow_target: np.ndarray) -> np.ndarray:
    """Pull toward the followed sheep with constant speed flocking_strength."""
    if cfg.flocking_strength == 0.0 or cfg.n_sheep == 1:
        return np.zeros_like(sheep)
    toward, _ = _unit(sheep[follow_target] - sheep)
    return cfg.flocking_strength * toward


def step_state(cfg: SheepdogConfig, state: State, action: np.ndarray, rng: np.random.Generator) -> State:
    """Advance one step. action: (n_dogs, 2) in [-1, 1], a velocity command for each dog."""
    L, dt = cfg.arena_size, cfg.dt

    # Dogs: velocity control, clipped at the walls.
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(cfg.n_dogs, 2), -1.0, 1.0)
    dogs = np.clip(state.dogs + action * cfg.dog_max_speed * dt, 0.0, L)

    # Sheep: re-pick follow targets whose timer ran out.
    timer = state.follow_timer - 1
    switch = timer <= 0
    follow_target = np.where(switch, sample_follow_targets(cfg.n_sheep, rng), state.follow_target)
    timer = np.where(switch, cfg.follow_switch_steps, timer)

    # Sheep: flee + flocking + noise, smoothed by inertia, capped at max speed.
    desired = (
        flee_velocity(cfg, state.sheep, dogs)
        + flock_velocity(cfg, state.sheep, follow_target)
        + cfg.noise_std * rng.standard_normal(state.sheep.shape)
    )
    vel = cfg.sheep_inertia * state.sheep_vel + (1.0 - cfg.sheep_inertia) * desired
    direction, speed = _unit(vel)
    vel = direction * np.minimum(speed, cfg.sheep_max_speed)[:, None]
    sheep = np.clip(state.sheep + vel * dt, 0.0, L)

    return replace(
        state,
        dogs=dogs,
        sheep=sheep,
        sheep_vel=(sheep - state.sheep) / dt,  # actual motion, so wall contact zeroes it
        follow_target=follow_target,
        follow_timer=timer,
        t=state.t + 1,
    )


def sheep_goal_dist(state: State) -> np.ndarray:
    return np.linalg.norm(state.sheep - state.goal, axis=-1)


def is_success(cfg: SheepdogConfig, state: State) -> bool:
    return bool((sheep_goal_dist(state) <= cfg.goal_radius).all())
