import json

from model.gpt import GPT
from train.grid import GRIDS, Grid

TABLE_1_ALPHA = {32: 3.37, 64: 3.39, 128: 3.53, 256: 3.51}


def test_synthetic_grid_is_table_1_four_layer_row():
    """The synthetic models are the paper's 4-layer Table 1 configurations, smallest first."""
    models = [(n_layer, d) for n_layer, d, _, _ in GRIDS["synthetic"].models]
    assert models == [(4, 32), (4, 64), (4, 128), (4, 256)]
    assert len(GRIDS["synthetic"].jobs()) == 21


def test_synthetic_sizes_bracket_expected_capacity():
    """Each model gets 2 dataset sizes below its expected ceiling and at least 3 above it."""
    for n_layer, d, _, sizes in GRIDS["synthetic"].models:
        ceiling = TABLE_1_ALPHA[d] * GPT(n_layer, d, vocab_size=2048).n_params() / 704
        assert sum(n < ceiling for n in sizes) == 2
        assert sum(n > ceiling for n in sizes) >= 3


def test_fineweb_grid():
    """The text grid is widths 64 and 128 on 2^12-2^17 at batch 256, lr 1e-3."""
    grid = GRIDS["fineweb"]
    assert [(n_layer, d, mb) for n_layer, d, mb, _ in grid.models] == [(8, 64, 128), (8, 128, 128)]
    sizes = [n for _, _, _, n in grid.jobs()]
    assert sizes == [2**k for k in range(12, 18)] * 2
    assert grid.batch_size == 256
    assert grid.args[grid.args.index("--lr") + 1] == "1e-3"


def test_fineweb_runs_get_equal_passes(tmp_path):
    """Every text run trains for the same number of passes over its data (7.5k steps at 4K)."""
    commands = GRIDS["fineweb"].run(tmp_path, dry_run=True)
    steps = {int(c[c.index("--n-train") + 1]): int(c[c.index("--max-steps") + 1]) for c in commands}
    assert steps == {4096: 7500, 8192: 15000, 16384: 30000, 32768: 60000,
                     65536: 120000, 131072: 240000}
    assert all(c[c.index("--batch-size") + 1] == "256" for c in commands)


def test_synthetic_runs_keep_shared_step_cap(tmp_path):
    """Without epochs, runs use the grid's shared --max-steps and no per-run budget."""
    commands = GRIDS["synthetic"].run(tmp_path, dry_run=True)
    assert all(c.count("--max-steps") == 1 for c in commands)
    assert all(c[c.index("--max-steps") + 1] == "20000" for c in commands)


def test_wide_models_use_micro_batches():
    """Widths 128 and 256 split each batch so their activations stay far below 16 GB."""
    micro = {d: mb for _, d, mb, _ in GRIDS["synthetic"].models}
    assert micro[128] <= 512 and micro[256] <= 512


def test_command_passes_micro_batch_size(tmp_path):
    """Each run is launched with its model's micro-batch size."""
    command = Grid("synthetic", [(1, 32, 128, [16])]).run(tmp_path, dry_run=True)[0]
    assert command[command.index("--micro-batch-size") + 1] == "128"


def test_dry_run_skips_finished_runs(tmp_path):
    """Runs that already have results.json are not run again."""
    grid = Grid("synthetic", [(1, 32, 2048, [16, 32])])
    done = tmp_path / "synthetic_L1_d32_n16_s0"
    done.mkdir()
    (done / "results.json").write_text("{}")
    commands = grid.run(tmp_path, dry_run=True)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--n-train") + 1] == "32"


def test_grid_trains_then_resumes(tmp_path):
    """A real tiny grid trains its runs, and running it again does nothing."""
    args = ["--max-steps", "2", "--eval-every", "1", "--n-test", "8"]
    grid = Grid("synthetic", [(1, 32, 2048, [16])], args)
    assert len(grid.run(tmp_path)) == 1
    results = json.loads((tmp_path / "synthetic_L1_d32_n16_s0" / "results.json").read_text())
    assert results["steps"] == 2
    assert grid.run(tmp_path) == []


def test_failed_run_does_not_stop_grid(tmp_path):
    """A run that crashes is reported and the grid moves on to the next one."""
    grid = Grid("synthetic", [(1, 32, 2048, [16, 32])], ["--max-steps", "not-a-number"])
    assert len(grid.run(tmp_path)) == 2
