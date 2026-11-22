# How Much Do Small Language Models Memorize?

Scaled-down replication of [Morris et al. (2025)](https://arxiv.org/abs/2505.24832)

```bash
uv sync
uv run pytest
```

## Reproducing the results

**Section 3: capacity on synthetic data**

```bash
uv run python -m train.grid synthetic
uv run python -m eval.capacity
```

**Section 4: memorization and double descent on FineWeb**

```bash
uv run python -m data.fineweb --n-sequences 150000
uv run python -m eval.memorization
uv run python -m train.grid fineweb --out results/long --save-model
uv run python -m eval.text --results results/long --out figures/long
```

**Section 5: membership inference**

```bash
uv run python -m eval.membership --data synthetic
uv run python -m eval.membership --data fineweb --results results/long --out figures/long
```

**Rare words**

```bash
uv run python -m eval.tfidf
```

**Extraction**

Uses the width-64 checkpoints saved by the FineWeb grid above.

```bash
uv run python -m eval.extraction --results results/long --d-model 64
```

Figures and summary tables are written to `figures/`.

## Deviations from the paper

- Mixed precision (fp32 weights, bf16 forward). Pure bf16 weights stopped updating during training.
- Synthetic: lr 1e-3 (not given in the paper), at most 20k steps with a plateau stop, 1 seed.
- Text: batch 256 and lr 1e-3 instead of 1e-4, with every dataset size trained for the same number of passes over its data (~469). Training sets go up to 131K sequences (paper: 16.8M).
- Extraction: width-64 model only, checked on 10K training and 10K test sequences per run.
