import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.fineweb import FineWebData
from data.synthetic import SyntheticData
from eval.memorization import total_memorization, unintended_memorization
from model.gpt import GPT


class Trainer:
    def __init__(
        self, model, train, test, lr=1e-4, batch_size=2048, micro_batch_size=2048,
        eval_size=10_000, seed=0, autocast=True,
    ):
        """Set up Adam with the training and test sequences on the model's device."""
        self.model = model
        self.autocast = autocast
        self.device = next(model.parameters()).device
        self.train = torch.as_tensor(train, device=self.device)
        self.test = torch.as_tensor(test, device=self.device)
        self.train_eval = self.train[:eval_size]
        self.batch_size = min(batch_size, len(self.train))
        self.micro_batch_size = min(micro_batch_size, self.batch_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.step = 0
        self.history = []
        self.converged = False

    def batches(self):
        """Yield batches of training indices, reshuffling every epoch."""
        n = len(self.train)
        while True:
            perm = torch.randperm(n, device=self.device, generator=self.generator)
            for i in range(0, n - self.batch_size + 1, self.batch_size):
                yield perm[i : i + self.batch_size]

    def train_step(self, idx):
        """Take one Adam step on a batch, accumulating gradients over micro-batches."""
        self.optimizer.zero_grad(set_to_none=True)
        for chunk in idx.split(self.micro_batch_size):
            x = self.train[chunk]
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.autocast):
                logits = self.model(x)
            loss = F.cross_entropy(logits.float().transpose(1, 2), x)
            (loss * len(chunk) / len(idx)).backward()
        self.optimizer.step()
        self.step += 1

    def fit(self, max_steps, eval_every=500, patience=10, tol=0.1):
        """Train until train bits per sequence improve by less than tol for patience evals."""
        best, stale, start = np.inf, 0, time.time()
        best_state, best_bits = None, np.inf
        for idx in self.batches():
            self.train_step(idx)
            if self.step % eval_every == 0 or self.step == max_steps:
                train_bits = self.model.bits(self.train_eval, self.micro_batch_size).mean()
                test_bits = self.model.bits(self.test, self.micro_batch_size).mean()
                self.history.append(
                    {
                        "step": self.step,
                        "train_bits": float(train_bits),
                        "test_bits": float(test_bits),
                        "seconds": time.time() - start,
                    }
                )
                print(f"step {self.step}: train {train_bits:.2f} bits, test {test_bits:.2f} bits")
                if train_bits < best_bits:
                    best_bits = train_bits
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if train_bits < best - tol:
                    best, stale = train_bits, 0
                else:
                    stale += 1
                self.converged = stale >= patience
                if self.converged or self.step >= max_steps:
                    break
        self.model.load_state_dict(best_state)
        return self.history


def run_name(data, n_layer, d_model, n_train, seed):
    """Return the results folder name for one run."""
    return f"{data}_L{n_layer}_d{d_model}_n{n_train}_s{seed}"


def main(argv=None):
    """Train one model on one dataset and save its results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "fineweb"], required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.1)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--pool", default="data/fineweb.npy")
    parser.add_argument("--reference-bits", default="data/fineweb_reference_bits.npy")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    if args.lr is None:
        args.lr = 1e-3 if args.data == "synthetic" else 1e-4

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data == "synthetic":
        data = SyntheticData(args.n_train, n_test=args.n_test, seed=args.seed)
        vocab_size, bos_id = data.vocab_size, 0
    else:
        data = FineWebData(args.n_train, n_test=args.n_test, seed=args.seed, path=args.pool)
        vocab_size, bos_id = FineWebData.vocab_size, 50256
    model = GPT(args.n_layer, args.d_model, vocab_size, bos_id=bos_id).to(device)

    trainer = Trainer(
        model, data.train, data.test, lr=args.lr, batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size, seed=args.seed,
    )
    history = trainer.fit(args.max_steps, args.eval_every, args.patience, args.tol)
    train_bits = model.bits(data.train, trainer.micro_batch_size)
    test_bits = model.bits(data.test, trainer.micro_batch_size)

    results = {
        "config": vars(args),
        "n_params": model.n_params(),
        "dtype": str(next(model.parameters()).dtype),
        "steps": trainer.step,
        "converged": trainer.converged,
        "history": history,
        "train_bits_mean": float(train_bits.mean()),
        "test_bits_mean": float(test_bits.mean()),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
    }
    if args.data == "synthetic":
        results["memorization"] = float(total_memorization(train_bits, data.entropy_bits()))
    elif Path(args.reference_bits).exists():
        reference_bits = np.load(args.reference_bits)[data.train_idx]
        results["memorization"] = float(unintended_memorization(train_bits, reference_bits))

    out = Path(args.out) / run_name(args.data, args.n_layer, args.d_model, args.n_train, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.save(out / "train_bits.npy", train_bits)
    np.save(out / "test_bits.npy", test_bits)
    print(f"saved {out}: {results.get('memorization', float('nan')):.0f} bits memorized")
    return results


if __name__ == "__main__":
    main()
