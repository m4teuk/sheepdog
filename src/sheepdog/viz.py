"""Opt-in visualisation: record an episode, then plot it (PNG) or animate it (GIF).

Needs the `viz` extra (matplotlib). Nothing here is called by the env itself.

    traj = record(env, policy, seed=0)
    plot_trajectory(traj, env.cfg, "episode.png")
    animate(traj, env.cfg, "episode.gif")
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless cluster: always render to files

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .config import SheepdogConfig  # noqa: E402
from .env import SheepdogEnv  # noqa: E402

SHEEP_COLOR = "#9a9a9a"
GOAL_COLOR = "#2ca02c"
DOG_COLORS = plt.get_cmap("tab10").colors[3:]  # red, purple, brown, ...


def record(
    env: SheepdogEnv,
    policy: Callable[[dict], np.ndarray] | None = None,
    seed: int | None = None,
    options: dict | None = None,
    max_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Run one episode. policy(obs) -> action; None means uniform random actions.

    Returns stacked observations, each with a leading time axis of length T + 1,
    plus "success" (bool) and "length" (T).
    """
    obs, info = env.reset(seed=seed, options=options)
    if policy is None:
        env.action_space.seed(seed)
        policy = lambda _: env.action_space.sample()  # noqa: E731
    frames = [obs]
    for _ in range(max_steps or env.cfg.max_episode_steps):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        frames.append(obs)
        if terminated or truncated:
            break
    traj = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    traj["success"] = np.bool_(info["success"])
    traj["length"] = np.int64(len(frames) - 1)
    return traj


def _setup_axes(ax, cfg: SheepdogConfig, goal: np.ndarray):
    L = cfg.arena_size
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Circle(goal, cfg.goal_radius, color=GOAL_COLOR, alpha=0.25, lw=0))
    ax.plot(*goal, "x", color=GOAL_COLOR, ms=8, mew=2)


def plot_trajectory(traj: dict, cfg: SheepdogConfig, path: str | Path | None = None, ax=None):
    """Static plot of a whole episode: paths, hollow start markers, filled end markers."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))
    _setup_axes(ax, cfg, traj["goal"][0])
    for i in range(traj["sheep"].shape[1]):
        p = traj["sheep"][:, i]
        ax.plot(p[:, 0], p[:, 1], color=SHEEP_COLOR, lw=1)
        ax.plot(*p[0], "o", mfc="none", mec=SHEEP_COLOR, ms=7)
        ax.plot(*p[-1], "o", color="white", mec="black", ms=8)
    for i in range(traj["dogs"].shape[1]):
        p, c = traj["dogs"][:, i], DOG_COLORS[i % len(DOG_COLORS)]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1, alpha=0.8)
        ax.plot(*p[0], "s", mfc="none", mec=c, ms=7)
        ax.plot(*p[-1], "s", color=c, ms=8)
    status = "success" if traj["success"] else "no success"
    ax.set_title(f"{int(traj['length'])} steps, {status}")
    if own_fig:
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=120)
            plt.close(fig)
    return ax


def animate(
    traj: dict,
    cfg: SheepdogConfig,
    path: str | Path,
    fps: int = 20,
    stride: int = 1,
    trail: int = 30,
) -> None:
    """Save the episode as a GIF. stride skips frames; trail is the tail length in steps.
    Dashed circles show each dog's flee radius."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _setup_axes(ax, cfg, traj["goal"][0])
    n_dogs, n_sheep = traj["dogs"].shape[1], traj["sheep"].shape[1]

    sheep_trails = [ax.plot([], [], color=SHEEP_COLOR, lw=1)[0] for _ in range(n_sheep)]
    dog_trails = [ax.plot([], [], color=DOG_COLORS[i % len(DOG_COLORS)], lw=1, alpha=0.6)[0] for i in range(n_dogs)]
    sheep_dots = ax.scatter(*traj["sheep"][0].T, s=60, c="white", edgecolors="black", zorder=3)
    dog_colors = [DOG_COLORS[i % len(DOG_COLORS)] for i in range(n_dogs)]
    dog_dots = ax.scatter(*traj["dogs"][0].T, s=70, marker="s", c=dog_colors, zorder=4)
    flee_circles = [
        ax.add_patch(plt.Circle((0, 0), cfg.flee_radius, fill=False, ls="--", lw=0.8, color=DOG_COLORS[i % len(DOG_COLORS)], alpha=0.5))
        for i in range(n_dogs)
    ]
    title = ax.set_title("")
    fig.tight_layout()

    T = int(traj["length"])
    frames = list(range(0, T + 1, stride))
    if frames[-1] != T:
        frames.append(T)

    def update(t):
        lo = max(0, t - trail)
        for i, line in enumerate(sheep_trails):
            line.set_data(traj["sheep"][lo : t + 1, i, 0], traj["sheep"][lo : t + 1, i, 1])
        for i, line in enumerate(dog_trails):
            line.set_data(traj["dogs"][lo : t + 1, i, 0], traj["dogs"][lo : t + 1, i, 1])
        sheep_dots.set_offsets(traj["sheep"][t])
        dog_dots.set_offsets(traj["dogs"][t])
        for i, circle in enumerate(flee_circles):
            circle.center = traj["dogs"][t, i]
        done = " (success)" if t == T and traj["success"] else ""
        title.set_text(f"step {t}/{T}{done}")
        return [*sheep_trails, *dog_trails, sheep_dots, dog_dots, *flee_circles, title]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
