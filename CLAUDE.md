# sheepdog

2D continuous herding env (Gymnasium API, OGBench-style): dogs (agent-controlled) push sheep to a goal point.
Full state observed, no rendering in the env. Run tests with `uv run --extra viz pytest`.

- `src/sheepdog/config.py`: `SheepdogConfig`, all parameters.
- `src/sheepdog/dynamics.py`: pure simulation core `(cfg, state, action, rng) -> state`. Keep it
  pure and vectorized: it will be ported to JAX to run batched on the GPU.
- `src/sheepdog/env.py`: thin Gymnasium wrapper (`SheepdogEnv`, id `Sheepdog-v0`).
- `src/sheepdog/viz.py` (opt-in, `viz` extra): `record` an episode, then `plot_trajectory` (PNG) / `animate` (GIF).
  Demo: `uv run --extra viz scripts/visualize.py`; outputs go to `outputs/` (gitignored).

Design decisions:
- Planned research: unsupervised RL, later maybe GCRL / contrastive RL. The env does not do reward
  shaping; reward is only a success indicator (`info["success"]`). Don't add task rewards.
- Support multiple dogs and sheep everywhere, even though experiments start with 1 dog and 1 sheep.
- Flocking (`flocking_strength`) defaults to 0 (off).
