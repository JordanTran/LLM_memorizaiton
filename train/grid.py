import argparse
import subprocess
import sys
from pathlib import Path

from train.train import run_name

ROOT = Path(__file__).resolve().parents[1]


class Grid:
    def __init__(self, data, models, args=()):
        """A set of runs: every (n_layer, d_model, micro_batch_size) model on each dataset size."""
        self.data = data
        self.models = models
        self.args = list(args)

    def jobs(self):
        """Return (n_layer, d_model, micro_batch_size, n_train) for every run, in order."""
        return [(n_layer, d, mb, n) for n_layer, d, mb, sizes in self.models for n in sizes]

    def run(self, out="results", seed=0, dry_run=False):
        """Train every run without saved results, one process each; return the commands run."""
        commands = []
        for n_layer, d_model, micro_batch_size, n_train in self.jobs():
            name = run_name(self.data, n_layer, d_model, n_train, seed)
            if (Path(out) / name / "results.json").exists():
                print(f"skip {name}: already done")
                continue
            command = [
                sys.executable, "-m", "train.train", "--data", self.data,
                "--n-layer", str(n_layer), "--d-model", str(d_model), "--n-train", str(n_train),
                "--micro-batch-size", str(micro_batch_size), "--seed", str(seed),
                "--out", str(out), *self.args,
            ]
            commands.append(command)
            print(" ".join(command[1:]), flush=True)
            if not dry_run and subprocess.run(command, cwd=ROOT).returncode != 0:
                print(f"FAILED {name}, continuing", flush=True)
        return commands


GRIDS = {
    "synthetic": Grid(
        "synthetic",
        [
            (4, 32, 2048, [256, 512, 1024, 2048, 4096]),
            (4, 64, 2048, [512, 1024, 2048, 4096, 8192]),
            (4, 128, 512, [2048, 4096, 8192, 16384, 32768]),
            (4, 256, 512, [8192, 16384, 32768, 65536, 131072]),
        ],
        ["--max-steps", "20000", "--patience", "6"],
    ),
    "fineweb": Grid(
        "fineweb",
        [
            (8, 64, 64, [2**14, 2**15, 2**16, 2**17]),
            (8, 128, 64, [2**14, 2**15, 2**16, 2**17]),
        ],
        ["--max-steps", "20000", "--patience", "6"],
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", choices=GRIDS)
    parser.add_argument("--out", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    GRIDS[args.grid].run(args.out, args.seed, args.dry_run)
