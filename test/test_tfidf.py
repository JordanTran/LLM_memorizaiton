import json

import numpy as np
import pytest

from data.fineweb import FineWebData
from eval.tfidf import RareWords, ranks, tfidf, words


def test_words_are_lowercase_and_split_on_punctuation():
    """Words ignore case and punctuation."""
    assert words("The cat, the HAT!") == ["the", "cat", "the", "hat"]


def test_tfidf_matches_the_papers_formula():
    """A word's weight is log(|D| / its count in D), averaged over the document's words."""
    scores = tfidf([["a", "b"], ["a"], []])
    assert scores[0] == pytest.approx((np.log(3 / 2) + np.log(3 / 1)) / 2)
    assert scores[1] == pytest.approx(np.log(3 / 2))
    assert scores[2] == 0.0


def test_ranks():
    """Ranks order values from smallest to largest."""
    assert list(ranks(np.array([3.0, 1.0, 2.0]))) == [2, 0, 1]


def fake_run(tmp_path, n_train=8, n_test=5):
    """Write a tiny token pool, reference bits and one run's per-sequence bits."""
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 50, size=(20, 4)).astype(np.uint16)
    np.save(tmp_path / "pool.npy", pool)
    np.save(tmp_path / "reference.npy", np.arange(20, dtype=float) * 8)
    run = tmp_path / "run"
    run.mkdir()
    config = {"n_layer": 8, "d_model": 64, "n_train": n_train, "n_test": n_test, "seed": 0}
    (run / "results.json").write_text(json.dumps({"config": config}))
    np.save(run / "train_bits.npy", np.full(n_train, 4.0))
    return run


def test_memorization_lines_up_reference_and_model_bits(tmp_path):
    """Each training sequence gets its own reference bits minus model bits, per token."""
    run = fake_run(tmp_path)
    rare = RareWords(run, tmp_path / "pool.npy", tmp_path / "reference.npy",
                     decode=lambda tokens: " ".join(f"w{t}" for t in tokens))
    data = FineWebData(8, n_test=5, seed=0, path=tmp_path / "pool.npy")
    assert rare.memorization == pytest.approx((data.train_idx * 8.0 - 4.0) / 4)
    assert len(rare.tfidf) == 8


def test_summary_and_plot(tmp_path):
    """The summary flags the rarest sequences and the figure is saved."""
    run = fake_run(tmp_path)
    rare = RareWords(run, tmp_path / "pool.npy", tmp_path / "reference.npy",
                     decode=lambda tokens: " ".join(f"w{t}" for t in tokens))
    rare.tfidf = np.arange(8.0)
    rare.memorization = np.arange(8.0) * 2
    s = rare.summary(top=0.25)
    assert s["spearman"] == pytest.approx(1.0)
    assert s["n_rarest"] == 2 and s["rarest_memorization"] == pytest.approx(13.0)
    rare.plot(tmp_path / "fig9.png", highlight=3)
    assert (tmp_path / "fig9.png").exists()
