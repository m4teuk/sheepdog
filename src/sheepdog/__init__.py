import gymnasium as gym

from .config import SheepdogConfig
from .dynamics import State
from .env import SheepdogEnv

__all__ = ["SheepdogConfig", "SheepdogEnv", "State"]

# Time limit is handled inside the env (cfg.max_episode_steps), so no max_episode_steps here.
gym.register(id="Sheepdog-v0", entry_point="sheepdog.env:SheepdogEnv")
