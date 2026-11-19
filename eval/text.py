import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.capacity import INK, MUTED, SURFACE, nats_per_token, style, width_color  # noqa: E402

REFERENCE_BITS_PER_SEQUENCE = 64 * 4.274


class TextResults:
    def __init__(self, results_dir="results", alpha=3.30):
        """Load every finished FineWeb run; alpha is bits per parameter from Section 3."""
        self.alpha = alpha
        self.rows = []
        for path in sorted(Path(results_dir).glob("fineweb_*/results.json")):
            run = json.loads(path.read_text())
            config = run["config"]
            n = config["n_train"]
            self.rows.append({
                "n_layer": config["n_layer"],
                "d_model": config["d_model"],
                "n_params": run["n_params"],
                "n_train": n,
                "memorization": run.get("memorization", float("nan")),
                "train_bits": run["train_bits_mean"],
                "test_bits": run["test_bits_mean"],
                "best_test_bits": min(h["test_bits"] for h in run["history"]),
                "capacity_ratio": n * REFERENCE_BITS_PER_SEQUENCE / (alpha * run["n_params"]),
            })
        if not self.rows:
            raise ValueError(f"no fineweb results in {results_dir}")
        self.rows.sort(key=lambda r: (r["n_params"], r["n_train"]))

    def models(self):
        """Return (n_layer, d_model, n_params) for every model, smallest first."""
        found = {(r["n_layer"], r["d_model"], r["n_params"]) for r in self.rows}
        return sorted(found, key=lambda m: m[2])

    def curve(self, n_layer, d_model):
        """Return one model's runs sorted by dataset size."""
        return [r for r in self.rows if (r["n_layer"], r["d_model"]) == (n_layer, d_model)]

    def table(self):
        """Return a markdown table of memorization and losses for every run."""
        lines = [
            "| Layers | Width | Train sequences | Dataset ÷ capacity | Memorization (bits) "
            "| Train bits/seq | Test bits/seq | Best test bits/seq |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r['n_layer']} | {r['d_model']} | {r['n_train']:,} | {r['capacity_ratio']:.2f} "
                f"| {r['memorization']:.3g} | {r['train_bits']:.0f} | {r['test_bits']:.0f} "
                f"| {r['best_test_bits']:.0f} |"
            )
        return "\n".join(lines)

    def _lines(self, ax, x_key, y_key, linestyle="-", hollow=False, label=True, scale=1.0):
        """Draw one line per model of y_key (times scale) against x_key."""
        for n_layer, d, params in self.models():
            rows = self.curve(n_layer, d)
            color = width_color(d)
            ax.plot([r[x_key] for r in rows], [r[y_key] * scale for r in rows], "o",
                    linestyle=linestyle, color=color, linewidth=2, markersize=6,
                    markerfacecolor=SURFACE if hollow else color,
                    markeredgecolor=color if hollow else SURFACE, markeredgewidth=1.5)
            if label:
                ax.plot([], [], "-", color=color, linewidth=4,
                        label=f"{n_layer}L d={d} ({params / 1e6:.2f}M)")

    def plot_memorization(self, path):
        """Fig. 2: unintended memorization against dataset size, with the dataset's total bits."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        sizes = sorted({r["n_train"] for r in self.rows})
        xs = np.array([sizes[0] / 1.5, sizes[-1] * 1.5])
        ax.plot(xs, xs * REFERENCE_BITS_PER_SEQUENCE, color=MUTED, linestyle="--", linewidth=1.2,
                label="total bits (dataset size)")
        self._lines(ax, "n_train", "memorization")
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "Unintended memorization of text", "Training set size (number of datapoints)",
              "Unintended memorization (bits)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)

    def plot_losses(self, path):
        """Fig. 4: train (solid) and test (dashed) loss per token against dataset size."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        scale = nats_per_token(1.0)
        self._lines(ax, "n_train", "train_bits", "-", scale=scale)
        self._lines(ax, "n_train", "test_bits", "--", hollow=True, label=False, scale=scale)
        ax.plot([], [], "-", color=MUTED, linewidth=2, label="train")
        ax.plot([], [], "--", color=MUTED, linewidth=2, label="test")
        ax.axhline(nats_per_token(REFERENCE_BITS_PER_SEQUENCE), color=MUTED, linestyle=":",
                   linewidth=1.2, label="774M reference")
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "Train and test loss on text", "Training set size (number of datapoints)",
              "Loss (nats per token)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right", handlelength=3.5)
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)

    def plot_test_loss_vs_capacity(self, path):
        """Appendix: the paper's Fig. 3 axes applied to text, test bits per sequence."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        self._lines(ax, "capacity_ratio", "test_bits")
        ax.axvline(1.0, color=INK, linestyle="--", linewidth=1.5,
                   label="model capacity = dataset size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "Double descent on text",
              f"Dataset-to-capacity ratio (274 bits/seq, {self.alpha:.2f} bits/param)",
              "Test loss (bits per sequence)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(argv=None):
    """Make the Section 4 figures and table from the FineWeb results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--alpha", type=float, default=3.30)
    args = parser.parse_args(argv)
    text = TextResults(args.results, args.alpha)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text.plot_memorization(out / "fig2_text_memorization.png")
    text.plot_losses(out / "fig4_text_losses.png")
    text.plot_test_loss_vs_capacity(out / "appendix_text_test_loss_vs_capacity.png")
    table = text.table()
    (out / "text_summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return text.rows


if __name__ == "__main__":
    main()
