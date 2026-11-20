import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.fineweb import FineWebData  # noqa: E402
from eval.capacity import INK, SURFACE, style, width_color  # noqa: E402

HIGHLIGHT = "#e07b39"


def words(text):
    """Return the lowercase words of a text."""
    return re.findall(r"\w+", text.lower())


def tfidf(docs):
    """Paper Sec. 4.1: mean over a document's words of log(|D| / count of the word in D)."""
    counts = Counter(w for doc in docs for w in doc)
    return np.array([
        np.mean([np.log(len(docs) / counts[w]) for w in doc]) if doc else 0.0 for doc in docs
    ])


def gpt2_decode(tokens):
    """Turn GPT-2 token ids back into text."""
    import tiktoken

    return tiktoken.get_encoding("gpt2").decode(tokens)


def ranks(x):
    """Return the rank of each value, 0 for the smallest."""
    out = np.empty(len(x))
    out[np.argsort(x)] = np.arange(len(x))
    return out


class RareWords:
    def __init__(self, run_dir, pool="data/fineweb.npy",
                 reference="data/fineweb_reference_bits.npy", decode=gpt2_decode):
        """Score one text run's training sequences by memorization and TF-IDF."""
        run_dir = Path(run_dir)
        self.config = json.loads((run_dir / "results.json").read_text())["config"]
        data = FineWebData(self.config["n_train"], n_test=self.config["n_test"],
                           seed=self.config["seed"], path=pool)
        model_bits = np.load(run_dir / "train_bits.npy")
        reference_bits = np.load(reference)[data.train_idx]
        self.memorization = (reference_bits - model_bits) / data.train.shape[1]
        self.tfidf = tfidf([words(decode(seq.tolist())) for seq in data.train])

    def summary(self, top=0.01):
        """Return the rank correlation and the memorization of the rarest-word sequences."""
        k = max(1, int(len(self.tfidf) * top))
        rarest = np.argsort(self.tfidf)[-k:]
        return {
            "spearman": float(np.corrcoef(ranks(self.tfidf), ranks(self.memorization))[0, 1]),
            "mean_memorization": float(self.memorization.mean()),
            "rarest_memorization": float(self.memorization[rarest].mean()),
            "n_rarest": k,
        }

    def plot(self, path, highlight=20):
        """Fig. 9: TF-IDF against memorization for every training sequence."""
        c = self.config
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        ax.scatter(self.memorization, self.tfidf, s=4, color=width_color(c["d_model"]),
                   alpha=0.5, linewidths=0, label="training sequences")
        top = np.argsort(self.tfidf)[-highlight:]
        ax.scatter(self.memorization[top], self.tfidf[top], s=18, color=HIGHLIGHT,
                   edgecolors=SURFACE, linewidths=0.5, label=f"{highlight} highest TF-IDF")
        ax.set_yscale("log")
        style(ax, f"Memorization vs. TF-IDF ({c['n_layer']}L d={c['d_model']}, "
              f"N = {c['n_train']:,})", "Kolmogorov memorization (bits per token)", "TF-IDF")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(argv=None):
    """Make Fig. 9 and a short summary for one text run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/long/fineweb_L8_d64_n65536_s0")
    parser.add_argument("--pool", default="data/fineweb.npy")
    parser.add_argument("--reference", default="data/fineweb_reference_bits.npy")
    parser.add_argument("--out", default="figures/long")
    args = parser.parse_args(argv)
    words_ = RareWords(args.run, args.pool, args.reference)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    words_.plot(out / "fig9_tfidf_vs_memorization.png")
    s = words_.summary()
    text = (
        f"Run: {args.run}\n"
        f"Spearman correlation of TF-IDF and memorization: {s['spearman']:.3f}\n"
        f"Mean memorization: {s['mean_memorization']:.3f} bits/token\n"
        f"Top {s['n_rarest']} TF-IDF sequences: {s['rarest_memorization']:.3f} bits/token\n"
    )
    (out / "tfidf_summary.md").write_text(text, encoding="utf-8")
    print(text)
    return s


if __name__ == "__main__":
    main()
