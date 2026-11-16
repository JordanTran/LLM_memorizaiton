import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BITS_PER_SEQUENCE = 704
TABLE_1 = {
    (1, 32): (4.23, 3.93), (1, 64): (3.92, 3.74), (1, 128): (3.65, 3.61), (1, 256): (3.12, 2.88),
    (2, 32): (4.16, 3.56), (2, 64): (4.08, 3.94), (2, 128): (3.99, 3.89), (2, 256): (4.01, 3.66),
    (4, 32): (3.92, 3.37), (4, 64): (3.98, 3.39), (4, 128): (3.78, 3.53), (4, 256): (3.68, 3.51),
    (8, 32): (3.02, 2.86), (8, 64): (3.83, 3.19), (8, 128): (3.89, 3.49), (8, 256): (3.96, 3.65),
}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
WIDTH_COLORS = {32: "#86b6ef", 64: "#3987e5", 128: "#1c5cab", 256: "#0d366b"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def ramp(n):
    """Return n light-to-dark steps of the validated blue ramp."""
    return [RAMP[round(i * (len(RAMP) - 1) / max(n - 1, 1))] for i in range(n)]


def width_color(d_model):
    """Return a model's fixed color, so it never changes as more results arrive."""
    return WIDTH_COLORS.get(d_model, MUTED)


def style(ax, title, xlabel, ylabel):
    """Apply recessive grid and axes, and set the titles."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


class Capacity:
    def __init__(self, results_dir="results"):
        """Load every finished synthetic run in the results folder."""
        paths = sorted(Path(results_dir).glob("synthetic_*/results.json"))
        self.runs = [json.loads(p.read_text()) for p in paths]
        if not self.runs:
            raise ValueError(f"no synthetic results in {results_dir}")

    def models(self):
        """Return (n_layer, d_model, n_params) for every model, smallest first."""
        found = {(r["config"]["n_layer"], r["config"]["d_model"], r["n_params"]) for r in self.runs}
        return sorted(found, key=lambda m: m[2])

    def curve(self, n_layer, d_model):
        """Return one model's runs sorted by dataset size."""
        runs = [
            r for r in self.runs
            if (r["config"]["n_layer"], r["config"]["d_model"]) == (n_layer, d_model)
        ]
        return sorted(runs, key=lambda r: r["config"]["n_train"])

    def capacity(self, n_layer, d_model):
        """Return a model's capacity: its most memorization over all dataset sizes."""
        return max(r["memorization"] for r in self.curve(n_layer, d_model))

    def peak_size(self, n_layer, d_model):
        """Return the dataset size at which a model memorized the most."""
        runs = self.curve(n_layer, d_model)
        return max(runs, key=lambda r: r["memorization"])["config"]["n_train"]

    def paper_alphas(self):
        """Return the paper's Table 1 bf16 mean and fitted bits/param for the models we ran."""
        params = np.array([p for _, _, p in self.models()], dtype=float)
        alpha = np.array([TABLE_1.get((n_layer, d), (np.nan, np.nan))[1]
                          for n_layer, d, _ in self.models()])
        caps = alpha * params
        return {"mean": float(alpha.mean()), "fit": float(params @ caps / (params @ params))}

    def alphas(self):
        """Return bits per parameter as the paper reports it: per model, mean, and fitted slope."""
        params = np.array([p for _, _, p in self.models()], dtype=float)
        caps = np.array([self.capacity(n_layer, d) for n_layer, d, _ in self.models()])
        per_model = caps / params
        return {
            "per_model": per_model.tolist(),
            "mean": float(per_model.mean()),
            "fit": float(params @ caps / (params @ params)),
            "log_log_slope": float(np.polyfit(np.log(params), np.log(caps), 1)[0])
            if len(params) > 1 else None,
        }

    def table(self):
        """Return a markdown table comparing each model's capacity with the paper's Table 1."""
        rows = [
            "| Layers | Width | Params | Runs (converged) | Capacity (bits) | Peak at "
            "| α ours | α paper bf16 | α paper fp32 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for (n_layer, d, params), alpha in zip(self.models(), self.alphas()["per_model"], strict=True):
            runs = self.curve(n_layer, d)
            converged = sum(r["converged"] for r in runs)
            fp32, bf16 = TABLE_1.get((n_layer, d), (float("nan"), float("nan")))
            peak = self.peak_size(n_layer, d)
            flag = " (largest tested)" if peak == runs[-1]["config"]["n_train"] else ""
            rows.append(
                f"| {n_layer} | {d} | {params:,} | {len(runs)} ({converged}) "
                f"| {self.capacity(n_layer, d):.3g} | {peak:,}{flag} "
                f"| {alpha:.2f} | {bf16:.2f} | {fp32:.2f} |"
            )
        return "\n".join(rows)

    def plot_memorization_vs_data(self, path):
        """Fig. 1: memorization against dataset size, one line per model."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        sizes = sorted({r["config"]["n_train"] for r in self.runs})
        xs = np.array([sizes[0] / 1.5, sizes[-1] * 1.5])
        ax.plot(xs, xs * BITS_PER_SEQUENCE, color=MUTED, linestyle="--", linewidth=1.2,
                label="dataset size in bits (everything memorized)")
        for n_layer, d, params in self.models():
            color = width_color(d)
            runs = self.curve(n_layer, d)
            n = [r["config"]["n_train"] for r in runs]
            mem = [r["memorization"] for r in runs]
            ax.plot(n, mem, color=color, linewidth=2, label=f"{n_layer}L d={d} ({params / 1e6:.2f}M)")
            for r, x, y in zip(runs, n, mem, strict=True):
                if r["converged"]:
                    ax.plot(x, y, "o", markersize=6, color=color, markeredgecolor=SURFACE,
                            markeredgewidth=1.5)
                else:
                    ax.plot(x, y, "o", markersize=7, markerfacecolor=SURFACE,
                            markeredgecolor=color, markeredgewidth=2)
            ax.annotate(f"d={d}", (n[-1], mem[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, "Memorization levels off at each model's capacity",
              "Training set size (sequences)", "Memorization (bits)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        fig.text(0.01, 0.01, "Hollow markers: run hit the step cap before converging.",
                 color=MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)

    def plot_capacity_vs_params(self, path):
        """Fig. 11: capacity against parameter count, with the fitted bits per parameter."""
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
        alphas = self.alphas()
        params = np.array([p for _, _, p in self.models()], dtype=float)
        xs = np.geomspace(params.min() / 1.5, params.max() * 1.5, 50)
        ax.plot(xs, alphas["fit"] * xs, color=MUTED, linewidth=1.2, linestyle="--",
                label=f"fit: {alphas['fit']:.2f} bits/param")
        ax.plot(xs, 3.64 * xs, color=MUTED, linewidth=1, linestyle=":",
                label="paper Fig. 11: 3.64 bits/param")
        for n_layer, d, p in self.models():
            color = width_color(d)
            cap = self.capacity(n_layer, d)
            ax.plot(p, cap, "o", markersize=8, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=f"{n_layer}L d={d}")
            ax.annotate(f"{cap / p:.2f}", (p, cap), xytext=(8, -4), textcoords="offset points",
                        color=INK, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "Capacity grows linearly with parameters", "Parameters", "Capacity (bits)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)

    def plot_memorization_vs_steps(self, path, n_layer, d_model):
        """Fig. 10: memorization during training for each dataset size of one model."""
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        runs = self.curve(n_layer, d_model)
        for r, color in zip(runs, ramp(len(runs)), strict=True):
            n = r["config"]["n_train"]
            steps = [h["step"] for h in r["history"]]
            mem = [n * (BITS_PER_SEQUENCE - h["train_bits"]) / 1e6 for h in r["history"]]
            ax.plot(steps, mem, color=color, linewidth=2, label=f"{n:,} sequences")
        ax.axhline(self.capacity(n_layer, d_model) / 1e6, color=MUTED, linestyle="--", linewidth=1.2,
                   label="capacity (best over sizes)")
        style(ax, f"Memorization during training ({n_layer}L d={d_model})",
              "Training step", "Memorization (million bits)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(argv=None):
    """Make the Section 3 figures and table from the synthetic results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)
    capacity = Capacity(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capacity.plot_memorization_vs_data(out / "fig1_memorization_vs_data.png")
    capacity.plot_capacity_vs_params(out / "fig11_capacity_vs_params.png")
    largest = capacity.models()[-1]
    capacity.plot_memorization_vs_steps(out / "fig10_memorization_vs_steps.png", *largest[:2])
    alphas = capacity.alphas()
    paper = capacity.paper_alphas()
    slope = alphas["log_log_slope"]
    slope = "n/a (needs 2+ models)" if slope is None else f"{slope:.3f}"
    summary = (
        f"{capacity.table()}\n\n"
        f"Mean bits/param: {alphas['mean']:.2f} "
        f"(paper Table 1 bf16, same models: {paper['mean']:.2f}; all 16 configs: 3.51)\n"
        f"Fitted bits/param (through origin): {alphas['fit']:.2f} "
        f"(paper, same models: {paper['fit']:.2f}; Fig. 11: 3.64)\n"
        "'(largest tested)': memorization was highest at the biggest dataset, "
        "so that model's capacity may be underestimated.\n"
        f"Log-log slope: {slope} (1.0 means capacity is linear in params)\n"
    )
    (out / "capacity_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return alphas


if __name__ == "__main__":
    main()
