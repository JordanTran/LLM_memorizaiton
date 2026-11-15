import copy
import json

import numpy as np
import torch

from data.fineweb import FineWebData
from data.synthetic import SyntheticData
from eval.memorization import total_memorization, unintended_memorization
from model.gpt import GPT
from train.train import Trainer, main

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_trainer(n=12, batch_size=4, micro_batch_size=4, lr=1e-3, autocast=False):
    """Return a Trainer for a small fp32 model on n synthetic sequences."""
    torch.manual_seed(0)
    data = SyntheticData(n, n_test=8)
    model = GPT(1, 32, 2048)
    return Trainer(
        model, data.train, data.test, lr, batch_size, micro_batch_size, autocast=autocast
    )


def test_overfit_memorizes_nearly_everything():
    """A mixed-precision model trained on 16 random sequences memorizes over 95% of their bits."""
    torch.manual_seed(0)
    data = SyntheticData(16, n_test=16)
    model = GPT(2, 128, 2048).to(DEVICE)
    Trainer(model, data.train, data.test, lr=1e-3).fit(max_steps=400, eval_every=100, patience=100)
    memorized = total_memorization(model.bits(data.train), data.entropy_bits())
    assert memorized > 0.95 * data.entropy_bits()


def test_mixed_precision_keeps_fp32_weights():
    """Training computes in bf16 but every weight stays fp32 and moves on every step."""
    trainer = tiny_trainer(autocast=True)
    dtypes = []
    trainer.model.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train_step(torch.arange(4))
    assert dtypes == [torch.bfloat16]
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
    assert all((p != b).all() for p, b in zip(trainer.model.parameters(), before, strict=True))


def test_epoch_visits_every_sequence_once():
    """Each epoch's batches cover every training index exactly once."""
    trainer = tiny_trainer(n=12, batch_size=4)
    batches = trainer.batches()
    seen = torch.cat([next(batches) for _ in range(3)])
    assert sorted(seen.tolist()) == list(range(12))


def test_each_epoch_reshuffles():
    """Consecutive epochs visit the training set in different orders."""
    trainer = tiny_trainer(n=12, batch_size=12)
    batches = trainer.batches()
    assert not torch.equal(next(batches), next(batches))


def test_batch_size_capped_at_dataset_size():
    """Datasets smaller than the batch size are trained with the full dataset as one batch."""
    trainer = tiny_trainer(n=10, batch_size=2048, micro_batch_size=2048)
    assert trainer.batch_size == 10 and trainer.micro_batch_size == 10
    assert len(next(trainer.batches())) == 10


def test_gradient_accumulation_matches_full_batch():
    """Splitting a batch into micro-batches gives the same update as one full batch."""
    full = tiny_trainer(n=8, batch_size=8, micro_batch_size=8)
    split = tiny_trainer(n=8, batch_size=8, micro_batch_size=2)
    split.model.load_state_dict(copy.deepcopy(full.model.state_dict()))
    idx = torch.arange(8)
    full.train_step(idx)
    split.train_step(idx)
    for a, b in zip(full.model.parameters(), split.model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_eval_uses_micro_batch_size(monkeypatch):
    """Evaluation scores in micro-batches, so it never needs more memory than training."""
    trainer = tiny_trainer(n=12, batch_size=12, micro_batch_size=3)
    sizes = []
    real_bits = trainer.model.bits
    monkeypatch.setattr(
        trainer.model, "bits", lambda x, batch_size=256: sizes.append(batch_size) or real_bits(x)
    )
    trainer.fit(max_steps=1, eval_every=1)
    assert sizes == [3, 3]


def test_stops_when_loss_plateaus():
    """With a zero learning rate, training stops after patience evals without improvement."""
    trainer = tiny_trainer(lr=0.0)
    history = trainer.fit(max_steps=10_000, eval_every=2, patience=3)
    assert trainer.step == 2 * (1 + 3)
    assert len(history) == 4


def test_stops_at_max_steps():
    """Training never runs past max_steps, and the last step is evaluated."""
    trainer = tiny_trainer()
    history = trainer.fit(max_steps=7, eval_every=5, patience=100)
    assert trainer.step == 7
    assert [h["step"] for h in history] == [5, 7]
    assert set(history[0]) == {"step", "train_bits", "test_bits", "seconds"}


def test_main_synthetic(tmp_path):
    """A synthetic run saves results with memorization and per-sequence bits."""
    args = "--data synthetic --n-train 32 --n-test 16 --n-layer 1 --d-model 32"
    args += f" --max-steps 4 --eval-every 2 --out {tmp_path}"
    results = main(args.split())
    out = tmp_path / "synthetic_L1_d32_n32_s0"
    saved = json.loads((out / "results.json").read_text())
    assert saved["memorization"] == results["memorization"]
    assert saved["steps"] == 4 and saved["n_params"] == GPT(1, 32, 2048).n_params()
    assert saved["dtype"] == "torch.float32"
    assert np.load(out / "train_bits.npy").shape == (32,)
    assert np.load(out / "test_bits.npy").shape == (16,)


def test_main_fineweb_uses_reference_bits(tmp_path):
    """A text run computes unintended memorization against the reference bits of its train rows."""
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 50256, size=(60, 64)).astype(np.uint16)
    reference = rng.uniform(2000, 4000, size=60)
    np.save(tmp_path / "pool.npy", pool)
    np.save(tmp_path / "ref.npy", reference)
    args = "--data fineweb --n-train 20 --n-test 10 --n-layer 1 --d-model 32"
    args += f" --max-steps 2 --eval-every 2 --out {tmp_path}"
    args += f" --pool {tmp_path / 'pool.npy'} --reference-bits {tmp_path / 'ref.npy'}"
    results = main(args.split())
    out = tmp_path / "fineweb_L1_d32_n20_s0"
    train_bits = np.load(out / "train_bits.npy")
    data = FineWebData(20, n_test=10, path=tmp_path / "pool.npy")
    expected = unintended_memorization(train_bits, reference[data.train_idx])
    assert expected > 0
    assert np.isclose(results["memorization"], expected)
