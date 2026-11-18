import json

import numpy as np
import pytest

from eval.membership import MembershipInference, attack, auc, best_threshold, f1, main


def test_best_threshold_separates_clean_groups():
    """With members always lower than non-members, the threshold falls between them."""
    bits = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    is_member = np.array([True, True, True, False, False, False])
    assert 3.0 < best_threshold(bits, is_member) < 10.0


def test_f1_known_values():
    """F1 matches a hand calculation: 2 of 3 guesses right, 2 of 4 members found."""
    bits = np.array([1.0, 2.0, 3.0, 8.0, 9.0, 4.0])
    is_member = np.array([True, True, False, True, True, False])
    precision, recall = 2 / 3, 2 / 4
    assert np.isclose(f1(bits, is_member, 3.5), 2 * precision * recall / (precision + recall))


def test_f1_zero_when_nothing_guessed():
    """Guessing no members gives F1 of zero, not a division error."""
    assert f1(np.array([5.0, 6.0]), np.array([True, False]), -np.inf) == 0.0


def test_auc_extremes():
    """AUC is 1 when members always have fewer bits, 0 when always more."""
    assert auc(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == 1.0
    assert auc(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 0.0


def test_separable_losses_give_perfect_attack():
    """When every member has lower loss than every non-member, F1 and AUC are 1."""
    rng = np.random.default_rng(0)
    result = attack(rng.uniform(0, 1, 5000), rng.uniform(2, 3, 5000))
    assert result["f1"] == 1.0 and result["auc"] == 1.0


def test_identical_distributions_give_chance():
    """With members and non-members from the same distribution, F1 and AUC are near 0.5."""
    rng = np.random.default_rng(0)
    result = attack(rng.normal(size=20_000), rng.normal(size=20_000))
    assert abs(result["auc"] - 0.5) < 0.02
    assert abs(result["f1"] - 0.5) < 0.1


def test_threshold_is_not_tuned_on_eval_half():
    """The threshold comes from the calibration half, so random data cannot score F1 near 2/3."""
    scores = [
        attack(np.random.default_rng(s).normal(size=4000),
               np.random.default_rng(s + 100).normal(size=4000), seed=s)["f1"]
        for s in range(10)
    ]
    assert np.mean(scores) < 0.6


def test_attack_balances_classes_and_caps_size():
    """The attack uses equal members and non-members, at most n of each."""
    assert attack(np.zeros(300), np.ones(50_000))["n_per_class"] == 300
    assert attack(np.zeros(50_000), np.ones(50_000), n=1000)["n_per_class"] == 1000


def write_run(root, data, d, n, train_bits, test_bits):
    """Save results.json and the per-sequence bits for one fake run."""
    folder = root / f"{data}_L4_d{d}_n{n}_s0"
    folder.mkdir(parents=True)
    run = {"config": {"n_layer": 4, "d_model": d, "n_train": n}, "n_params": d * 1000}
    (folder / "results.json").write_text(json.dumps(run))
    np.save(folder / "train_bits.npy", train_bits)
    np.save(folder / "test_bits.npy", test_bits)


def test_loads_one_dataset_type_sorted(tmp_path):
    """Only runs of the requested data type are attacked, sorted by model then dataset size."""
    rng = np.random.default_rng(0)
    for d, n in [(64, 512), (32, 1024), (32, 256)]:
        write_run(tmp_path, "synthetic", d, n, rng.normal(size=n), rng.normal(size=500))
    write_run(tmp_path, "fineweb", 32, 256, rng.normal(size=256), rng.normal(size=500))
    rows = MembershipInference(tmp_path, "synthetic").rows
    assert [(r["d_model"], r["n_train"]) for r in rows] == [(32, 256), (32, 1024), (64, 512)]


def test_no_results_fail_loudly(tmp_path):
    """Asking for a data type with no finished runs gives a clear error."""
    with pytest.raises(ValueError, match="no fineweb results"):
        MembershipInference(tmp_path, "fineweb")


def test_main_writes_figure_and_table(tmp_path):
    """The script saves the F1 plot and a UTF-8 table for the requested data type."""
    rng = np.random.default_rng(0)
    for n in (256, 1024):
        write_run(tmp_path / "results", "fineweb", 64, n, rng.normal(size=n) - 1,
                  rng.normal(size=500))
    main(["--data", "fineweb", "--results", str(tmp_path / "results"),
          "--out", str(tmp_path / "figures")])
    assert (tmp_path / "figures" / "membership_fineweb.png").exists()
    table = (tmp_path / "figures" / "membership_fineweb.md").read_bytes().decode("utf-8")
    assert "| 4 | 64 | 1,024 |" in table
