#!/usr/bin/env python3
"""
validate_i_mp.py
================
Compares the exact (non-sampling) i-MP against the sampling estimate,
with runtime for each, across the example pedigrees already used to
validate the per-individual match probability engine
(``Pedigrees/*.csv``).

  exact
      ``compute_at_least_one_match_exact`` -- the full inclusion-exclusion
      sum (equation 13), computed with *no sampling at all*. Feasible
      whenever the candidate count is small (``<= EXACT_MAX_CANDIDATES``);
      cost is ``O(2^|candidates|)``.
  sampled
      ``compute_at_least_one_match`` -- the exact-sampling estimator
      (Section 3.2), run with a large, fixed-seed sample size.

Usage
-----
    python validate_i_mp.py [--mu MU] [--n-samples N] [--seed SEED] [--out results.csv]
"""

import argparse
import csv
import glob
import os
import time

from ystr_pedigree import (
    DEFAULT_MU,
    compute_at_least_one_match,
    compute_at_least_one_match_exact,
    load_pedigree_from_csv,
)

PEDIGREE_DIR = os.path.join(os.path.dirname(__file__), "Pedigrees")
EXACT_MAX_CANDIDATES = 15


def validate(mu: float, n_samples: int, seed: int) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(PEDIGREE_DIR, "*.csv"))):
        name = os.path.basename(path)
        males, loci = load_pedigree_from_csv(path)

        t0 = time.perf_counter()
        sampled = compute_at_least_one_match(
            males, loci, mu=mu, n_samples=n_samples, seed=seed
        )
        t_sampled = time.perf_counter() - t0
        n_cand = len(sampled.candidates)

        exact = None
        t_exact = None
        if 0 < n_cand <= EXACT_MAX_CANDIDATES:
            t0 = time.perf_counter()
            exact = compute_at_least_one_match_exact(males, loci, mu=mu)
            t_exact = time.perf_counter() - t0

        rows.append(dict(
            pedigree=name, n=len(males), loci=len(loci), candidates=n_cand,
            exact=exact, t_exact=t_exact,
            sampled=sampled.i_mp, t_sampled=t_sampled,
        ))
    return rows


def _fmt_time(t: float | None) -> str:
    if t is None:
        return "n/a"
    return f"{t*1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def print_table(rows: list[dict]) -> None:
    cols = [
        ("pedigree", 22), ("n", 4), ("loci", 5), ("cand", 5),
        ("exact", 12), ("t_exact", 10),
        ("sampled", 12), ("t_sampled", 10),
    ]
    header = "".join(f"{c:>{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        exact_s = f"{r['exact']:.6f}" if r["exact"] is not None else "n/a"
        vals = [
            r["pedigree"], r["n"], r["loci"], r["candidates"],
            exact_s, _fmt_time(r["t_exact"]),
            f"{r['sampled']:.6f}", _fmt_time(r["t_sampled"]),
        ]
        print("".join(f"{v!s:>{w}}" for v, (_, w) in zip(vals, cols)))


def write_csv(rows: list[dict], out_path: str) -> None:
    fieldnames = ["pedigree", "n", "loci", "candidates", "exact", "t_exact", "sampled", "t_sampled"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fieldnames} for r in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--out", metavar="OUTPUT_CSV", default=None,
        help="Also write the table as CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = validate(args.mu, args.n_samples, args.seed)
    print_table(rows)
    if args.out:
        write_csv(rows, args.out)
        print(f"\nTable written to '{args.out}'")
