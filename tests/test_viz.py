import pytest

pytest.importorskip("matplotlib")

from sheepdog import SheepdogEnv  # noqa: E402
from sheepdog.viz import animate, plot_trajectory, record  # noqa: E402


def test_record_and_render(tmp_path):
    env = SheepdogEnv(n_dogs=2, n_sheep=3)
    traj = record(env, seed=0, max_steps=10)
    assert traj["sheep"].shape == (11, 3, 2)
    assert traj["dogs"].shape == (11, 2, 2)
    plot_trajectory(traj, env.cfg, tmp_path / "ep.png")
    animate(traj, env.cfg, tmp_path / "ep.gif", stride=3)
    assert (tmp_path / "ep.png").stat().st_size > 0
    assert (tmp_path / "ep.gif").stat().st_size > 0
