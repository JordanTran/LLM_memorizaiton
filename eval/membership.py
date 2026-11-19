import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.capacity import (  # noqa: E402
    BITS_PER_SEQUENCE, INK, MUTED, SURFACE, nats_per_token, style, width_color,
)
from eval.text import REFERENCE_BITS_PER_SEQUENCE  # noqa: E402


def best_threshold(bits, is_member):
    """Return the bits threshold that maximizes accuracy when guessing 'member' below it."""
    order = np.argsort(bits)
    sorted_bits, labels = bits[order], is_member[order]
    k = np.arange(len(bits) + 1)
    true_pos = np.concatenate([[0], np.cumsum(labels)])
    true_neg = (~is_member).sum() - (k - true_pos)
    best = int(np.argmax(true_pos + true_neg))
    if best == 0:
        return -np.inf
    if best == len(bits):
        return np.inf
    return (sorted_bits[best - 1] + sorted_bits[best]) / 2


def f1(bits, is_member, threshold):
    """Return the F1 score of guessing 'member' for sequences with bits below the threshold."""
    guess = bits < threshold
    true_pos = (guess & is_member).sum()
    if true_pos == 0:
        return 0.0
    precision = true_pos / guess.sum()
    recall = true_pos / is_member.sum()
    return float(2 * precision * recall / (precision + recall))


def auc(member_bits, nonmember_bits):
    """Return the probability that a random member has fewer bits than a random non-member."""
    bits = np.concatenate([member_bits, nonmember_bits])
    ranks = np.empty(len(bits))
    ranks[np.argsort(-bits)] = np.arange(1, len(bits) + 1)
    n_member, n_nonmember = len(member_bits), len(nonmember_bits)
    return float((ranks[:n_member].sum() - n_member * (n_member + 1) / 2) / (n_member * n_nonmember))


def attack(train_bits, test_bits, n=10_000, seed=0):
    """Loss-threshold attack on equal numbers of members and non-members.

    The threshold is chosen for accuracy on one half and F1 is reported on the other half,
    so an attack no better than chance scores about 0.5.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(train_bits), len(test_bits))
    members = rng.choice(train_bits, n, replace=False)
    nonmembers = rng.choice(test_bits, n, replace=False)
    half = n // 2
    calib_bits = np.concatenate([members[:half], nonmembers[:half]])
    calib_member = np.arange(2 * half) < half
    eval_bits = np.concatenate([members[half:], nonmembers[half:]])
    eval_member = np.arange(len(eval_bits)) < n - half
    threshold = best_threshold(calib_bits, calib_member)
    return {
        "n_per_class": n,
        "threshold": float(threshold),
        "f1": f1(eval_bits, eval_member, threshold),
        "auc": auc(members, nonmembers),
    }


class MembershipInference:
    def __init__(self, results_dir="results", data="synthetic", seed=0):
        """Run the attack on every finished run of one dataset type."""
        self.data = data
        self.results_dir = results_dir
        self.rows = []
        for path in sorted(Path(results_dir).glob(f"{data}_*/results.json")):
            run = json.loads(path.read_text())
            train_bits = np.load(path.parent / "train_bits.npy")
            test_bits = np.load(path.parent / "test_bits.npy")
            config = run["config"]
            self.rows.append({
                "n_layer": config["n_layer"],
                "d_model": config["d_model"],
                "n_params": run["n_params"],
                "n_train": config["n_train"],
                **attack(train_bits, test_bits, seed=seed),
            })
        if not self.rows:
            raise ValueError(f"no {data} results in {results_dir}")
        self.rows.sort(key=lambda r: (r["n_params"], r["n_train"]))

    def table(self):
        """Return a markdown table of attack F1 and AUC for every run."""
        lines = ["| Layers | Width | Train sequences | F1 | AUC |", "|---|---|---|---|---|"]
        for r in self.rows:
            lines.append(
                f"| {r['n_layer']} | {r['d_model']} | {r['n_train']:,} "
                f"| {r['f1']:.3f} | {r['auc']:.3f} |"
            )
        return "\n".join(lines)

    def models(self):
        """Return (n_layer, d_model, n_params) for every model, smallest first."""
        found = {(r["n_layer"], r["d_model"], r["n_params"]) for r in self.rows}
        return sorted(found, key=lambda m: m[2])

    def _plot_f1(self, path, x_key, xlabel, title):
        """Draw attack F1 against x_key, one line per model."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2, label="random guessing")
        for n_layer, d, params in self.models():
            rows = [r for r in self.rows if (r["n_layer"], r["d_model"]) == (n_layer, d)]
            x = [r[x_key] for r in rows]
            scores = [r["f1"] for r in rows]
            ax.plot(x, scores, "o-", color=width_color(d), linewidth=2, markersize=6,
                    markeredgecolor=SURFACE)
            ax.plot([], [], "-", color=width_color(d), linewidth=4,
                    label=f"{n_layer}L d={d} ({params / 1e6:.2f}M)")
            ax.annotate(f"d={d}", (x[-1], scores[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_xscale("log")
        ax.set_ylim(0.48, 1.02)
        style(ax, title, xlabel, "Membership inference score (F1)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left",
                  title="Model size", title_fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)

    def plot(self, path):
        """Figs. 7 and 15: attack F1 against training set size, one line per model."""
        self._plot_f1(path, "n_train", "Training set size (number of datapoints)",
                      f"Membership inference F1 across dataset sizes ({self.data})")

    def plot_vs_capacity(self, path, alpha=3.30):
        """Appendix: attack F1 against dataset bits ÷ (alpha × params)."""
        bits = BITS_PER_SEQUENCE if self.data == "synthetic" else REFERENCE_BITS_PER_SEQUENCE
        for r in self.rows:
            r["capacity_ratio"] = r["n_train"] * bits / (alpha * r["n_params"])
        self._plot_f1(path, "capacity_ratio", f"Dataset-to-capacity ratio ({alpha:.2f} bits/param)",
                      f"Membership inference F1 against dataset ÷ capacity ({self.data})")

    def plot_histogram(self, path, n_layer, d_model, n_train):
        """Fig. 8: counts of per-token train and test loss for one run."""
        pattern = f"{self.data}_L{n_layer}_d{d_model}_n{n_train}_s*"
        folder = next(Path(self.results_dir).glob(pattern))
        train = nats_per_token(np.load(folder / "train_bits.npy"))
        test = nats_per_token(np.load(folder / "test_bits.npy"))
        bins = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), 50)
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        ax.hist(train, bins=bins, color=width_color(64), alpha=0.75,
                label=f"train ({len(train):,})")
        ax.hist(test, bins=bins, color=width_color(256), alpha=0.6, label=f"test ({len(test):,})")
        ax.set_yscale("log")
        style(ax, f"Compression rate, {self.data} ({n_layer}L d={d_model}, N = {n_train:,})",
              "Loss (nats per token)", "Count")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(argv=None):
    """Make the membership inference figure and table for one dataset type."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "fineweb"], default="synthetic")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)
    mia = MembershipInference(args.results, args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mia.plot(out / f"membership_{args.data}.png")
    mia.plot_vs_capacity(out / f"appendix_membership_vs_capacity_{args.data}.png")
    run = next((r for r in mia.rows if r["d_model"] == 128 and r["n_train"] == 2**14), None)
    if run:
        mia.plot_histogram(out / f"fig8_loss_histogram_{args.data}.png", run["n_layer"], 128, 2**14)
    table = mia.table()
    (out / f"membership_{args.data}.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return mia.rows


if __name__ == "__main__":
    main()
