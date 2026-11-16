import json

import numpy as np
import pytest

from eval.capacity import TABLE_1, Capacity, main, width_color

PARAMS = {32: 118_464, 64: 335_232, 128: 1_063_680, 256: 3_700_224}


def write_run(root, d, n, memorization, converged=True, data="synthetic"):
    """Save a minimal results.json for one fake run."""
    folder = root / f"{data}_L4_d{d}_n{n}_s0"
    folder.mkdir(parents=True)
    history = [{"step": 500, "train_bits": 704 - memorization / n, "test_bits": 800.0}]
    run = {
        "config": {"n_layer": 4, "d_model": d, "n_train": n},
        "n_params": PARAMS[d],
        "converged": converged,
        "memorization": memorization,
        "history": history,
    }
    (folder / "results.json").write_text(json.dumps(run))


def fake_grid(root, alpha):
    """Write runs whose capacity is alpha(d) * params, reached at the middle dataset size."""
    for d in PARAMS:
        cap = alpha(d) * PARAMS[d]
        for n, frac in [(1024 * d, 0.5), (2048 * d, 1.0), (4096 * d, 0.97)]:
            write_run(root, d, n, cap * frac)


def test_capacity_is_max_over_dataset_sizes(tmp_path):
    """A model's capacity is its highest memorization across all dataset sizes."""
    write_run(tmp_path, 32, 256, 1e5)
    write_run(tmp_path, 32, 512, 4e5)
    write_run(tmp_path, 32, 1024, 3.8e5)
    assert Capacity(tmp_path).capacity(4, 32) == 4e5


def test_models_and_curves_are_sorted(tmp_path):
    """Models come smallest first, and each curve runs from smallest to largest dataset."""
    for d, n in [(256, 4096), (32, 512), (256, 1024), (32, 256)]:
        write_run(tmp_path, d, n, 1e5)
    capacity = Capacity(tmp_path)
    assert [m[1] for m in capacity.models()] == [32, 256]
    assert [r["config"]["n_train"] for r in capacity.curve(4, 256)] == [1024, 4096]


def test_constant_alpha_is_recovered(tmp_path):
    """If every model stores 3.5 bits per parameter, all three estimates say 3.5."""
    fake_grid(tmp_path, lambda d: 3.5)
    alphas = Capacity(tmp_path).alphas()
    assert np.allclose(alphas["per_model"], 3.5)
    assert np.isclose(alphas["mean"], 3.5) and np.isclose(alphas["fit"], 3.5)
    assert np.isclose(alphas["log_log_slope"], 1.0)


def test_reproduces_paper_table_1_numbers(tmp_path):
    """With the paper's own 4-layer capacities, we get its mean (3.45) and fit (3.51)."""
    fake_grid(tmp_path, lambda d: TABLE_1[(4, d)][1])
    alphas = Capacity(tmp_path).alphas()
    assert np.allclose(alphas["per_model"], [3.37, 3.39, 3.53, 3.51])
    assert round(alphas["mean"], 2) == 3.45
    assert round(alphas["fit"], 2) == 3.51


def test_fit_is_least_squares_through_origin(tmp_path):
    """The fitted slope matches numpy's least squares with no intercept."""
    fake_grid(tmp_path, lambda d: {32: 3.0, 64: 3.4, 128: 3.6, 256: 3.8}[d])
    capacity = Capacity(tmp_path)
    params = np.array([p for _, _, p in capacity.models()], dtype=float)
    caps = np.array([capacity.capacity(n_layer, d) for n_layer, d, _ in capacity.models()])
    expected = np.linalg.lstsq(params[:, None], caps, rcond=None)[0][0]
    assert np.isclose(capacity.alphas()["fit"], expected)


def test_table_reports_convergence_and_paper_values(tmp_path):
    """The table counts converged runs and lists the paper's alpha for each model."""
    write_run(tmp_path, 128, 4096, 3.5e6, converged=True)
    write_run(tmp_path, 128, 8192, 3.7e6, converged=False)
    row = Capacity(tmp_path).table().splitlines()[-1]
    assert "| 2 (1) |" in row
    assert "| 3.53 | 3.78 |" in row
    assert "| 8,192 (largest tested) |" in row


def test_peak_not_flagged_when_curve_turns_down(tmp_path):
    """A model that peaks before its largest dataset is not flagged."""
    write_run(tmp_path, 32, 512, 3.0e5)
    write_run(tmp_path, 32, 1024, 2.9e5)
    row = Capacity(tmp_path).table().splitlines()[-1]
    assert "| 512 |" in row and "largest tested" not in row


def test_paper_comparison_uses_only_models_present(tmp_path):
    """The paper's mean and fit are computed for the models we actually ran."""
    write_run(tmp_path, 32, 512, 3.0e5)
    write_run(tmp_path, 256, 8192, 1.2e7)
    paper = Capacity(tmp_path).paper_alphas()
    assert np.isclose(paper["mean"], (3.37 + 3.51) / 2)
    p = np.array([PARAMS[32], PARAMS[256]], dtype=float)
    assert np.isclose(paper["fit"], p @ (np.array([3.37, 3.51]) * p) / (p @ p))


def test_ignores_text_runs(tmp_path):
    """FineWeb results in the same folder are not mixed into the capacity analysis."""
    write_run(tmp_path, 32, 256, 1e5)
    write_run(tmp_path, 64, 256, 9e9, data="fineweb")
    assert [m[1] for m in Capacity(tmp_path).models()] == [32]


def test_empty_results_fail_loudly(tmp_path):
    """Running the analysis before any synthetic run finishes gives a clear error."""
    with pytest.raises(ValueError, match="no synthetic results"):
        Capacity(tmp_path)


def test_main_writes_figures_and_summary(tmp_path):
    """The script saves Figs. 1, 10 and 11 plus a summary with the paper comparison."""
    fake_grid(tmp_path / "results", lambda d: 3.5)
    main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "figures")])
    names = {p.name for p in (tmp_path / "figures").iterdir()}
    assert names == {
        "fig1_memorization_vs_data.png",
        "fig10_memorization_vs_steps.png",
        "fig11_capacity_vs_params.png",
        "capacity_summary.md",
    }
    summary = (tmp_path / "figures" / "capacity_summary.md").read_bytes().decode("utf-8")
    assert "α ours" in summary and "3.50" in summary
    assert "same models: 3.45" in summary and "same models: 3.51" in summary


def test_main_works_with_one_model(tmp_path):
    """Partial results with a single finished model still produce figures and a summary."""
    for n, mem in [(256, 1.8e5), (512, 3.0e5), (1024, 2.9e5)]:
        write_run(tmp_path / "results", 32, n, mem)
    main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "figures")])
    summary = (tmp_path / "figures" / "capacity_summary.md").read_bytes().decode("utf-8")
    assert "n/a (needs 2+ models)" in summary


def test_model_colors_are_fixed():
    """Each width keeps its color no matter which other models have finished."""
    colors = [width_color(d) for d in (32, 64, 128, 256)]
    assert len(set(colors)) == 4
