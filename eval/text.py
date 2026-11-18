import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.capacity import INK, MUTED, SURFACE, style, width_color  # noqa: E402

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

    def _lines(self, ax, x_key, y_key, linestyle="-", label_suffix="", hollow=False):
        """Draw one line per model of y_key against x_key."""
        for n_layer, d, params in self.models():
            rows = self.curve(n_layer, d)
            x = [r[x_key] for r in rows]
            y = [r[y_key] for r in rows]
            color = width_color(d)
            ax.plot(x, y, "o", linestyle=linestyle, color=color, linewidth=2, markersize=6,
                    markerfacecolor=SURFACE if hollow else color,
                    markeredgecolor=color if hollow else SURFACE, markeredgewidth=1.5,
                    label=f"{n_layer}L d={d} ({params / 1e6:.2f}M){label_suffix}")

    def plot_memorization(self, path):
        """Fig. 2: unintended memorization against dataset size, one line per model."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        self._lines(ax, "n_train", "memorization")
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1e4)
        ax.set_ylim(0, 2 * max(r["memorization"] for r in self.rows))
        style(ax, "Unintended memorization on FineWeb", "Training set size (sequences)",
              "Memorization beyond the 774M reference (bits)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)

    def plot_losses(self, path, x_key="n_train"):
        """Figs. 3 and 4: train (dashed) and test (solid) bits per sequence against data size."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        self._lines(ax, x_key, "test_bits", "-", " test")
        self._lines(ax, x_key, "train_bits", "--", " train", hollow=True)
        ax.axhline(REFERENCE_BITS_PER_SEQUENCE, color=MUTED, linestyle=":", linewidth=1.2,
                   label="774M reference")
        if x_key == "capacity_ratio":
            ax.axvline(1.0, color=MUTED, linewidth=1, label="dataset = capacity")
            ax.set_xscale("log")
            xlabel = "Dataset bits ÷ model capacity"
        else:
            ax.set_xscale("log", base=2)
            xlabel = "Training set size (sequences)"
        style(ax, "Train and test loss on FineWeb", xlabel, "Bits per sequence")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right", handlelength=3.5)
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
    text.plot_losses(out / "fig3_text_losses_vs_capacity.png", "capacity_ratio")
    table = text.table()
    (out / "text_summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return text.rows


if __name__ == "__main__":
    main()
