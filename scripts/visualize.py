"""Record one episode and save a trajectory plot (PNG) and an animation (GIF).

    uv run --extra viz scripts/visualize.py --policy herder --seed 0
    uv run --extra viz scripts/visualize.py --policy random --n-sheep 5 --flocking 0.5
"""

import argparse
from pathlib import Path

import numpy as np

from sheepdog import SheepdogEnv
from sheepdog.viz import animate, plot_trajectory, record


def scripted_herder(env: SheepdogEnv):
    """Baseline: dog 0 stands behind the sheep's centre of mass, on the side away from the goal."""
    cfg = env.cfg

    def policy(obs):
        sheep, goal, dog = obs["sheep"].mean(0), obs["goal"], obs["dogs"][0]
        away = (sheep - goal) / (np.linalg.norm(sheep - goal) + 1e-8)
        to_target = sheep + away * 0.6 * cfg.flee_radius - dog
        action = np.zeros((cfg.n_dogs, 2))
        action[0] = np.clip(2.0 * to_target / (np.linalg.norm(to_target) + 1e-8), -1, 1)
        return action

    return policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["random", "herder"], default="herder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sheep", type=int, default=1)
    p.add_argument("--flocking", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stride", type=int, default=2, help="animate every n-th step")
    p.add_argument("--out", type=Path, default=Path("outputs/viz"))
    args = p.parse_args()

    env = SheepdogEnv(n_sheep=args.n_sheep, flocking_strength=args.flocking)
    policy = scripted_herder(env) if args.policy == "herder" else None
    traj = record(env, policy, seed=args.seed, max_steps=args.max_steps)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.policy}_seed{args.seed}"
    plot_trajectory(traj, env.cfg, stem.with_suffix(".png"))
    animate(traj, env.cfg, stem.with_suffix(".gif"), stride=args.stride)
    print(f"{int(traj['length'])} steps, success={bool(traj['success'])} -> {stem}.png / .gif")


if __name__ == "__main__":
    main()
