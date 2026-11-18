import json

import numpy as np
import pytest

from eval.text import REFERENCE_BITS_PER_SEQUENCE, TextResults, main


def write_run(root, d, n, memorization, train_bits, test_history, data="fineweb"):
    """Save a minimal results.json for one fake text run."""
    folder = root / f"{data}_L8_d{d}_n{n}_s0"
    folder.mkdir(parents=True)
    run = {
        "config": {"n_layer": 8, "d_model": d, "n_train": n},
        "n_params": 1_000_000 * d // 64,
        "memorization": memorization,
        "train_bits_mean": train_bits,
        "test_bits_mean": test_history[-1],
        "history": [{"test_bits": t} for t in test_history],
    }
    (folder / "results.json").write_text(json.dumps(run))


def test_loads_text_runs_sorted_and_ignores_synthetic(tmp_path):
    """Only FineWeb runs are loaded, sorted by model size then dataset size."""
    write_run(tmp_path, 128, 1024, 1.0, 300.0, [600.0])
    write_run(tmp_path, 64, 2048, 1.0, 300.0, [600.0])
    write_run(tmp_path, 64, 1024, 1.0, 300.0, [600.0])
    write_run(tmp_path, 64, 512, 1.0, 300.0, [600.0], data="synthetic")
    rows = TextResults(tmp_path).rows
    assert [(r["d_model"], r["n_train"]) for r in rows] == [(64, 1024), (64, 2048), (128, 1024)]


def test_best_test_is_minimum_over_training(tmp_path):
    """The best test loss is the lowest seen during training, not the final one."""
    write_run(tmp_path, 64, 1024, 1.0, 300.0, [650.0, 540.0, 900.0])
    row = TextResults(tmp_path).rows[0]
    assert row["best_test_bits"] == 540.0 and row["test_bits"] == 900.0


def test_capacity_ratio(tmp_path):
    """Dataset bits over capacity uses the reference's bits per sequence and alpha x params."""
    write_run(tmp_path, 64, 4096, 1.0, 300.0, [600.0])
    row = TextResults(tmp_path, alpha=2.0).rows[0]
    assert np.isclose(row["capacity_ratio"], 4096 * REFERENCE_BITS_PER_SEQUENCE / (2.0 * 1e6))


def test_no_results_fail_loudly(tmp_path):
    """Running before any text run finishes gives a clear error."""
    with pytest.raises(ValueError, match="no fineweb results"):
        TextResults(tmp_path)


def test_main_writes_figures_and_table(tmp_path):
    """The script saves Figs. 2, 3 and 4 and a UTF-8 table."""
    for n, mem, train, test in [(1024, 1e6, 200.0, 1800.0), (2048, 2e5, 330.0, 880.0)]:
        write_run(tmp_path / "results", 64, n, mem, train, [600.0, test])
    main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "figures")])
    names = {p.name for p in (tmp_path / "figures").iterdir()}
    assert names == {
        "fig2_text_memorization.png",
        "fig3_text_losses_vs_capacity.png",
        "fig4_text_losses.png",
        "text_summary.md",
    }
    table = (tmp_path / "figures" / "text_summary.md").read_bytes().decode("utf-8")
    assert "Dataset ÷ capacity" in table and "| 8 | 64 | 2,048 |" in table
