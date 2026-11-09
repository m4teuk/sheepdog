# sheepdog

2D continuous herding env (Gymnasium API, OGBench-style): dogs (agent-controlled) push sheep to a goal point.
Full state observed, no rendering. Run tests with `uv run pytest`.

- `src/sheepdog/config.py`: `SheepdogConfig`, all parameters.
- `src/sheepdog/dynamics.py`: pure simulation core `(cfg, state, action, rng) -> state`. Keep it
  pure and vectorized: it will be ported to JAX to run batched on the GPU.
- `src/sheepdog/env.py`: thin Gymnasium wrapper (`SheepdogEnv`, id `Sheepdog-v0`).

Design decisions:
- Planned research: unsupervised RL, later maybe GCRL / contrastive RL. The env does not do reward
  shaping; reward is only a success indicator (`info["success"]`). Don't add task rewards.
- Support multiple dogs and sheep everywhere, even though experiments start with 1 dog and 1 sheep.
- Flocking (`flocking_strength`) defaults to 0 (off).
