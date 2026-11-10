from dataclasses import dataclass


@dataclass(frozen=True)
class SheepdogConfig:
    """All environment parameters. Distances in arena units, speeds in units per second."""

    # World
    arena_size: float = 10.0  # square arena [0, arena_size]^2 with walls
    dt: float = 0.1
    max_episode_steps: int = 500

    # Entities
    n_dogs: int = 1
    n_sheep: int = 1

    # Dog
    dog_max_speed: float = 2.0  # action in [-1, 1]^2 is scaled by this

    # Sheep
    sheep_max_speed: float = 1.5
    flee_radius: float = 2.5  # sheep react to dogs closer than this
    flee_strength: float = 2.0  # flee speed when the dog is right next to the sheep (falls off linearly)
    noise_std: float = 0.3  # std of the Gaussian random-walk term
    sheep_inertia: float = 0.5  # in [0, 1): fraction of last step's velocity kept
    flocking_strength: float = 0.0  # pull toward the currently followed sheep; 0 disables flocking
    follow_switch_steps: int = 20  # each sheep picks a new sheep to follow this often

    # Goal
    goal_radius: float = 1.0  # success when every sheep is within this distance of the goal
    min_start_sheep_goal_dist: float = 1.5  # keeps episodes from starting solved

    def __post_init__(self):
        if self.n_dogs < 1 or self.n_sheep < 1:
            raise ValueError("need at least one dog and one sheep")
        if not 0.0 <= self.sheep_inertia < 1.0:
            raise ValueError("sheep_inertia must be in [0, 1)")
        if self.follow_switch_steps < 1:
            raise ValueError("follow_switch_steps must be >= 1")
        if self.min_start_sheep_goal_dist >= self.arena_size:
            raise ValueError("min_start_sheep_goal_dist must be smaller than arena_size")
