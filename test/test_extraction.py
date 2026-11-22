import json

import numpy as np
import torch

from eval.extraction import Extraction, greedy_extract, load_run, main
from model.gpt import GPT
from train.train import main as train_main


class CountingModel(torch.nn.Module):
    """Stub with the GPT interface that always predicts the previous token plus one."""

    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Logits at position t predict token t from the tokens before it."""
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return torch.nn.functional.one_hot((prev + 1) % self.vocab_size, self.vocab_size).float()

    def next_logits(self, x):
        """Logits for the token after x."""
        return self(torch.cat([x, torch.zeros_like(x[:, :1])], dim=1))[:, -1]


def test_greedy_extract_matches_only_sequences_the_model_predicts():
    """A sequence counts as extracted only if every token after the prefix is reproduced."""
    counting = np.array([[3, 4, 5, 6, 7, 8]])
    broken_late = np.array([[3, 4, 5, 6, 7, 0]])
    broken_middle = np.array([[3, 4, 0, 6, 7, 8]])
    broken_in_prefix = np.array([[3, 9, 0, 1, 2, 3]])
    seqs = np.concatenate([counting, broken_late, broken_middle, broken_in_prefix])
    hits = greedy_extract(CountingModel(), seqs, prefix_len=2, batch_size=3)
    assert list(hits) == [True, False, False, True]


def test_greedy_extract_uses_the_whole_prefix():
    """With a prefix as long as the sequence there is nothing to generate, so every row matches."""
    seqs = np.random.default_rng(0).integers(0, 10, size=(5, 6))
    assert greedy_extract(CountingModel(), seqs, prefix_len=6).all()


def test_next_logits_matches_full_forward():
    """next_logits(x) equals the full forward's logits at the position after x."""
    torch.manual_seed(0)
    model = GPT(2, 32, 50, bos_id=7).eval()
    x = torch.randint(0, 50, (3, 10))
    full = model(torch.cat([x, torch.zeros_like(x[:, :1])], dim=1))[:, -1]
    assert torch.allclose(model.next_logits(x), full, atol=1e-5)


def test_save_model_round_trips(tmp_path):
    """A run saved with --save-model reloads into the same model and the same splits."""
    args = "--data synthetic --n-train 32 --n-test 16 --n-layer 1 --d-model 32"
    args += f" --max-steps 4 --eval-every 2 --out {tmp_path} --save-model"
    train_main(args.split())
    run = tmp_path / "synthetic_L1_d32_n32_s0"
    model, data, config = load_run(run)
    assert config["n_train"] == 32 and data.train.shape == (32, 64)
    bits = model.bits(data.train)
    assert np.allclose(bits, np.load(run / "train_bits.npy"), atol=1e-3)


def test_main_writes_figure_table_and_json(tmp_path):
    """The script measures every saved run at prefixes 8, 16 and 32 and saves its outputs."""
    args = "--data synthetic --n-train 16 --n-test 8 --n-layer 1 --d-model 32"
    args += f" --max-steps 2 --eval-every 2 --out {tmp_path / 'results'} --save-model"
    train_main(args.split())
    main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "figures"), "--n", "8"])
    names = {p.name for p in (tmp_path / "figures").iterdir()}
    assert names == {"fig6_extraction_rates.png", "extraction_summary.md", "extraction.json"}
    rows = json.loads((tmp_path / "figures" / "extraction.json").read_text())
    assert [r["prefix"] for r in rows] == [8, 16, 32]
    assert all(0 <= r["train_rate"] <= 1 and r["n_train_checked"] == 8 for r in rows)


def test_d_model_keeps_only_runs_of_that_width(tmp_path):
    """With d_model set, saved runs of other widths are skipped."""
    for d in (32, 64):
        args = f"--data synthetic --n-train 16 --n-test 8 --n-layer 1 --d-model {d}"
        args += f" --max-steps 2 --eval-every 2 --out {tmp_path} --save-model"
        train_main(args.split())
    rows = Extraction(tmp_path, n=8, d_model=32).rows
    assert len(rows) == 3 and all(r["d_model"] == 32 for r in rows)
    assert {r["d_model"] for r in Extraction(tmp_path, n=8).rows} == {32, 64}


def test_runs_without_saved_models_fail_loudly(tmp_path):
    """Pointing the script at runs without model.pt is an error, not an empty figure."""
    (tmp_path / "run").mkdir()
    try:
        Extraction(tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
