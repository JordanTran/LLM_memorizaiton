import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.fineweb import FineWebData  # noqa: E402
from data.synthetic import SyntheticData  # noqa: E402
from eval.capacity import INK, MUTED, SURFACE, style  # noqa: E402
from model.gpt import GPT  # noqa: E402

PREFIXES = (8, 16, 32)
PREFIX_COLORS = {8: "#86b6ef", 16: "#3987e5", 32: "#0d366b"}


@torch.no_grad()
def greedy_extract(model, seqs, prefix_len, batch_size=1024):
    """Return, for each sequence, whether greedy decoding from its prefix reproduces the rest."""
    device = next(model.parameters()).device
    hits = []
    for i in range(0, len(seqs), batch_size):
        x = torch.as_tensor(seqs[i:i + batch_size], device=device)
        out = x[:, :prefix_len]
        for _ in range(prefix_len, x.shape[1]):
            next_token = model.next_logits(out).argmax(-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        hits.append((out == x).all(dim=1).cpu().numpy())
    return np.concatenate(hits)


def load_run(run_dir, pool="data/fineweb.npy", device="cpu"):
    """Rebuild one saved run's model and its exact train and test splits."""
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    c = results["config"]
    if c["data"] == "synthetic":
        data = SyntheticData(c["n_train"], n_test=c["n_test"], seed=c["seed"])
        vocab_size, bos_id = data.vocab_size, 0
    else:
        data = FineWebData(c["n_train"], n_test=c["n_test"], seed=c["seed"], path=pool)
        vocab_size, bos_id = FineWebData.vocab_size, 50256
    model = GPT(c["n_layer"], c["d_model"], vocab_size, bos_id=bos_id)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return model.to(device).eval(), data, c


class Extraction:
    def __init__(self, results_dir, pool="data/fineweb.npy", n=10_000, seed=0, d_model=None):
        """Measure train and test extraction rates for every run that saved its model.

        With d_model set, only runs of that width are measured.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(seed)
        self.rows = []
        for model_path in sorted(Path(results_dir).glob("*/model.pt")):
            config = json.loads((model_path.parent / "results.json").read_text())["config"]
            if d_model is not None and config["d_model"] != d_model:
                continue
            model, data, c = load_run(model_path.parent, pool, device)
            train = data.train[rng.permutation(len(data.train))[:n]]
            test = data.test[:n]
            for k in PREFIXES:
                self.rows.append({
                    "data": c["data"], "n_layer": c["n_layer"], "d_model": c["d_model"],
                    "n_train": c["n_train"], "prefix": k,
                    "train_rate": float(greedy_extract(model, train, k).mean()),
                    "test_rate": float(greedy_extract(model, test, k).mean()),
                    "n_train_checked": len(train), "n_test_checked": len(test),
                })
        if not self.rows:
            raise FileNotFoundError(f"no model.pt files under {results_dir}")

    def table(self):
        """Return a markdown table of extraction rates."""
        lines = ["| Model | Train size | Prefix | Train extraction | Test extraction |",
                 "|---|---|---|---|---|"]
        for r in sorted(self.rows, key=lambda r: (r["d_model"], r["n_train"], r["prefix"])):
            lines.append(f"| {r['n_layer']}L d={r['d_model']} | {r['n_train']:,} | {r['prefix']} "
                         f"| {r['train_rate']:.4f} | {r['test_rate']:.4f} |")
        return "\n".join(lines) + "\n"

    def plot(self, path):
        """Fig. 6: train (solid) and test (dashed) extraction rate against dataset size."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        for k in PREFIXES:
            rows = sorted([r for r in self.rows if r["prefix"] == k], key=lambda r: r["n_train"])
            n = np.array([r["n_train"] for r in rows])
            for key, style_ in (("train_rate", "-o"), ("test_rate", "--o")):
                y = np.array([r[key] for r in rows])
                keep = y > 0
                ax.plot(n[keep], y[keep], style_, color=PREFIX_COLORS[k], linewidth=2,
                        markersize=6, markerfacecolor=SURFACE if key == "test_rate" else None,
                        markeredgecolor=PREFIX_COLORS[k])
            ax.plot([], [], "-", color=PREFIX_COLORS[k], linewidth=4, label=f"prefix {k}")
        ax.plot([], [], "-", color=MUTED, linewidth=2, label="train")
        ax.plot([], [], "--", color=MUTED, linewidth=2, label="test")
        ax.set_xscale("log")
        if any(r["train_rate"] > 0 or r["test_rate"] > 0 for r in self.rows):
            ax.set_yscale("log")
        style(ax, "Extraction rates of 64-token sequences",
              "Training set size (number of datapoints)", "Extraction rate")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", handlelength=3.5)
        fig.text(0.01, 0.01, "Rates of exactly 0 are not drawn on the log axis; see the table.",
                 color=MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(argv=None):
    """Make Fig. 6 and a table for every run under --results that saved its model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/extraction")
    parser.add_argument("--pool", default="data/fineweb.npy")
    parser.add_argument("--out", default="figures/extraction")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=None)
    args = parser.parse_args(argv)
    extraction = Extraction(args.results, args.pool, args.n, d_model=args.d_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extraction.plot(out / "fig6_extraction_rates.png")
    table = extraction.table()
    (out / "extraction_summary.md").write_text(table, encoding="utf-8")
    (out / "extraction.json").write_text(json.dumps(extraction.rows, indent=2))
    print(table)
    return extraction


if __name__ == "__main__":
    main()
