#!/usr/bin/env python3
"""Independent verification of RAG chunking eval results.

Reads results/results.json (a list of per-question records) and:
  1. Recomputes the per-strategy summary table from scratch.
  2. Asserts data invariants, failing loudly (non-zero exit) on violation.
  3. Checks predicted-vs-actual mean-recall strategy ordering.

Stdlib + json only. Does NOT import rag_chunking_eval, so it verifies the
pipeline instead of repeating it.

Usage:
    python scripts/analyze_results.py [--results results/results.json --k 5
        --expected "hierarchical-250-1000,semantic-0.5,fixed-500-100,fixed-500-0"]
"""

import argparse
import json
import math
import sys

DEFAULT_EXPECTED = "hierarchical-250-1000,semantic-0.5,fixed-500-100,fixed-500-0"


def die(name, detail):
    """Report a broken invariant and exit non-zero."""
    print(f"INVARIANT FAILED [{name}]: {detail}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Verify rag-chunking-eval results.")
    ap.add_argument("--results", default="results/results.json",
                    help="Path to results JSON (list of per-question records).")
    ap.add_argument("--k", type=int, default=5,
                    help="Expected number of retrieved_ids per record.")
    ap.add_argument("--expected", default=DEFAULT_EXPECTED,
                    help="Comma-separated strategy names, best to worst, for ordering check.")
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        records = json.load(f)

    # The run script wraps records in an envelope: {"run": {...}, "records": [...]}.
    if isinstance(records, dict) and "records" in records:
        records = records["records"]

    if not isinstance(records, list) or not records:
        die("results_not_a_nonempty_list",
            f"expected a non-empty JSON list (or envelope with 'records') in {args.results}")

    required = {"question_id", "strategy", "retrieved_ids", "recall", "answer",
                "n_claims", "n_supported", "faithfulness"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            die("record_missing_keys",
                f"record {i} missing keys: {sorted(missing)}")

    # ---------- Invariant checks ----------
    strategies = {}
    for i, rec in enumerate(records):
        qid, strat = rec["question_id"], rec["strategy"]
        strategies.setdefault(strat, []).append((i, rec))

        # recall in [0,1]
        r = rec["recall"]
        if not isinstance(r, (int, float)) or isinstance(r, bool) \
                or not (0.0 <= r <= 1.0):
            die("recall_out_of_range",
                f"record {i} (q={qid}, s={strat}): recall={r!r} not in [0,1]")

        # retrieved_ids has exactly k entries, except hierarchical: its
        # expand step dedupes children that share a parent, so 1..k is valid.
        rids = rec["retrieved_ids"]
        n = len(rids) if isinstance(rids, list) else -1
        ok = (n == args.k) or (strat.startswith("hierarchical") and 1 <= n <= args.k)
        if not isinstance(rids, list) or not ok:
            die("retrieved_ids_wrong_length",
                f"record {i} (q={qid}, s={strat}): {n} ids, expected k={args.k}"
                + (" (1..k allowed for hierarchical)" if strat.startswith("hierarchical") else ""))

        # n_supported <= n_claims wherever both present
        nc, ns = rec["n_claims"], rec["n_supported"]
        if nc is not None and ns is not None:
            if not (isinstance(nc, int) and isinstance(ns, int)):
                die("claim_counts_not_int",
                    f"record {i} (q={qid}, s={strat}): "
                    f"n_claims={nc!r}, n_supported={ns!r}")
            if ns > nc:
                die("n_supported_gt_n_claims",
                    f"record {i} (q={qid}, s={strat}): "
                    f"n_supported={ns} > n_claims={nc}")

        # faithfulness == n_supported / n_claims within 1e-9 wherever both present
        f = rec["faithfulness"]
        if f is not None and nc is not None and ns is not None:
            if not isinstance(f, (int, float)) or isinstance(f, bool):
                die("faithfulness_not_number",
                    f"record {i} (q={qid}, s={strat}): faithfulness={f!r}")
            if nc == 0:
                die("faithfulness_zero_claims",
                    f"record {i} (q={qid}, s={strat}): "
                    f"faithfulness present but n_claims=0")
            if not math.isclose(f, ns / nc, rel_tol=0.0, abs_tol=1e-9):
                die("faithfulness_arithmetic_mismatch",
                    f"record {i} (q={qid}, s={strat}): faithfulness={f} != "
                    f"n_supported/n_claims={ns}/{nc}={ns / nc}")

    # every strategy has the same n
    counts = {s: len(v) for s, v in strategies.items()}
    if len(set(counts.values())) != 1:
        die("strategy_n_mismatch", f"per-strategy counts differ: {counts}")

    # no duplicate question_id within a strategy
    for s, v in strategies.items():
        qids = [rec["question_id"] for _, rec in v]
        if len(set(qids)) != len(qids):
            die("duplicate_question_id",
                f"strategy {s}: duplicate question_id(s) "
                f"{sorted({q for q in qids if qids.count(q) > 1})}")

    # ---------- Summary table ----------
    print("# RAG chunking eval: results summary\n")
    print("| strategy | n | mean recall | mean faithfulness | empty answers |")
    print("|---|---|---|---|---|")

    stats = {}
    for s, v in sorted(strategies.items()):
        n = len(v)
        mean_recall = sum(rec["recall"] for _, rec in v) / n
        f_vals = [rec["faithfulness"] for _, rec in v
                  if rec["faithfulness"] is not None]
        mean_faith = (sum(f_vals) / len(f_vals)) if f_vals else float("nan")
        n_empty = sum(1 for _, rec in v if rec["faithfulness"] is None)
        stats[s] = (n, mean_recall, mean_faith, n_empty, len(f_vals))
        faith_str = f"{mean_faith:.4f}" if f_vals else "n/a"
        print(f"| {s} | {n} | {mean_recall:.4f} | {faith_str} | {n_empty} |")

    print(f"\n(all invariants passed; k={args.k}; faithfulness mean excludes null records)\n")

    # ---------- Predicted vs actual ordering ----------
    expected = [e.strip() for e in args.expected.split(",") if e.strip()]
    actual = sorted(stats.keys(), key=lambda s: stats[s][1], reverse=True)
    print("## Predicted vs actual (mean recall, best to worst)\n")
    print(f"Expected: {', '.join(expected) if expected else '(none given)'}")
    print(f"Actual:   {', '.join(actual)}")
    match = actual == [s for s in expected if s in stats]
    if not expected:
        print("Result: no expected order supplied; check skipped.")
    elif match:
        print("Result: MATCH — actual ordering equals expected ordering.")
    else:
        print("Result: MISMATCH — actual ordering differs from expected ordering.")


if __name__ == "__main__":
    main()
