#!/usr/bin/env python3
"""Plot RAG chunking experiment results.

Reads results/summary.csv (one row per chunking strategy) and writes two bar
charts into the same results directory:

    recall.png        mean_recall per strategy (Recall@5)
    faithfulness.png  mean_faithfulness per strategy (averaged over
                      non-empty answers only)

Expected CSV columns (read via csv.DictReader):
    strategy, n, mean_recall, mean_faithfulness, n_empty_answers

Usage:
    python scripts/make_plots.py [--csv results/summary.csv --outdir results/]

Exits non-zero with a clear message if the input CSV is missing or a required
column is absent.
"""

from __future__ import annotations

import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed display order for the bar charts (documented contract).
STRATEGY_ORDER = [
    "fixed-500-0",
    "fixed-500-100",
    "semantic-0.5",
    "hierarchical-250-1000",
]

REQUIRED_COLUMNS = {
    "strategy",
    "n",
    "mean_recall",
    "mean_faithfulness",
    "n_empty_answers",
}

# The run script writes friendlier column names; accept both.
COLUMN_ALIASES = {
    "recall@5": "mean_recall",
    "faithfulness": "mean_faithfulness",
}


def load_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"error: '{csv_path}' has no header row.")
        reader.fieldnames = [
            COLUMN_ALIASES.get(name, name) for name in reader.fieldnames
        ]
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise SystemExit(
                f"error: '{csv_path}' is missing required column(s): "
                + ", ".join(missing)
            )
        return list(reader)


def ordered_strategies(rows: list[dict[str, str]]) -> list[str]:
    present = [r["strategy"] for r in rows]
    # Known strategies in the documented fixed order, then any unexpected
    # strategies appended in CSV order so nothing silently disappears.
    ordered = [s for s in STRATEGY_ORDER if s in present]
    ordered += [s for s in present if s not in STRATEGY_ORDER]
    return ordered


def bar_chart(
    path: str,
    title: str,
    subtitle: str,
    strategies: list[str],
    values: dict[str, float],
    n_label: str,
) -> None:
    vals = [values[s] for s in strategies]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = range(len(strategies))
    bars = ax.bar(xs, vals, color="#1f77b4", edgecolor="none", width=0.62)
    ax.set_ylim(0, 1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("score", fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8.5, color="#555555")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(v + 0.015, 0.985),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path} ({n_label})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot RAG chunking eval results from summary.csv."
    )
    parser.add_argument("--csv", default="results/summary.csv",
                        help="input CSV path (default: results/summary.csv)")
    parser.add_argument("--outdir", default="results/",
                        help="output directory (default: results/)")
    args = parser.parse_args(argv)

    try:
        with open(args.csv, encoding="utf-8"):
            pass
    except FileNotFoundError:
        print(f"error: input CSV not found: {args.csv}", file=sys.stderr)
        return 1

    rows = load_rows(args.csv)
    if not rows:
        print(f"error: '{args.csv}' contains no data rows.", file=sys.stderr)
        return 1

    strategies = ordered_strategies(rows)
    try:
        recall = {r["strategy"]: float(r["mean_recall"]) for r in rows}
        faith = {r["strategy"]: float(r["mean_faithfulness"]) for r in rows}
    except ValueError as exc:
        print(f"error: non-numeric score in '{args.csv}': {exc}",
              file=sys.stderr)
        return 1

    n_vals = {r["n"] for r in rows}
    n_label = n_vals.pop() if len(n_vals) == 1 else "mixed-n"
    n_tag = f"n={n_label}"

    import os
    os.makedirs(args.outdir, exist_ok=True)
    outdir = args.outdir.rstrip("/") + "/"

    bar_chart(
        outdir + "recall.png",
        f"Recall@5 by chunking strategy ({n_tag}, SQuAD v2 dev slice)",
        "Mean recall over all evaluated questions; higher is better.",
        strategies,
        recall,
        n_tag,
    )
    bar_chart(
        outdir + "faithfulness.png",
        f"Faithfulness by chunking strategy ({n_tag}, SQuAD v2 dev slice)",
        "Averaged over non-empty answers only; higher is better.",
        strategies,
        faith,
        n_tag,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
