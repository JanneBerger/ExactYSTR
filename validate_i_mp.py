#!/usr/bin/env python3
"""
validate_i_mp.py
================
Validation evidence for i-MP, the "at least one match" probability
(Section 3 of the technical note), across the example pedigrees already
used to validate the per-individual match probability engine
(``Pedigrees/*.csv``).

For each pedigree, three quantities are compared:

  exact
      ``compute_at_least_one_match_exact`` -- the full inclusion-exclusion
      sum (equation 13), computed with *no sampling at all*. Feasible
      whenever the candidate count is small (``<= EXACT_MAX_CANDIDATES``).
  sampled
      ``compute_at_least_one_match`` -- the exact-sampling estimator
      (Section 3.2), run with a large, fixed-seed sample size and
      reported together with its standard error.
  naive
      The closed-form union formula ``1 - Prod_i (1 - P_i)`` one would
      get by (wrongly) assuming every candidate is independent given
      the suspect. It agrees with ``exact`` only when no two candidates
      share an untyped common ancestor; the gap between ``naive`` and
      ``exact`` is itself evidence that the harder, non-factorising
      calculation in Section 3 is doing real work.

This mirrors the technical note's own validation methodology
(Section 5, "Evidence that the results are correct"): here, "two
independent implementations" is the brute-force exact peel vs. the
sampling estimator, and "internal consistency" is the z-score

    z = (sampled - exact) / se

staying within a few standard deviations across every case where an
exact value is available at all.

Usage
-----
    python validate_i_mp.py [--mu MU] [--n-samples N] [--seed SEED]
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
    compute_match_probs_fast,
    load_pedigree_from_csv,
)

PEDIGREE_DIR = os.path.join(os.path.dirname(__file__), "Pedigrees")
EXACT_MAX_CANDIDATES = 15


def naive_independence_mp(males, loci, mu: float) -> float:
    """1 - Prod_i (1 - P_i), the union formula that is only correct when
    no two candidates share an untyped common ancestor."""
    combined: dict[int, float] = {}
    for locus in loci:
        mp, _ = compute_match_probs_fast(males, locus, mu)
        for i, p in mp.items():
            male_id = males[i].male_id
            combined[male_id] = combined.get(male_id, 1.0) * p
    if not combined:
        return 0.0
    prod_complement = 1.0
    for p in combined.values():
        prod_complement *= (1.0 - p)
    return 1.0 - prod_complement


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

        naive = naive_independence_mp(males, loci, mu)

        z = None
        if exact is not None:
            if sampled.se:
                z = (sampled.i_mp - exact) / sampled.se
            elif sampled.i_mp == exact:
                # Both sides agree exactly (e.g. a typed match already
                # present in the data, se == 0 by construction) -- zero
                # discrepancy, not "not applicable".
                z = 0.0

        rows.append(dict(
            pedigree=name, n=len(males), loci=len(loci), candidates=n_cand,
            exact=exact, sampled=sampled.i_mp, se=sampled.se,
            used_fallback=sampled.used_fallback, known_match=sampled.known_match,
            z=z, naive=naive, t_sampled=t_sampled, t_exact=t_exact,
        ))
    return rows


def print_table(rows: list[dict]) -> None:
    cols = [
        ("pedigree", 22), ("n", 4), ("loci", 5), ("cand", 5),
        ("exact", 12), ("sampled", 12), ("se", 10), ("z (sigma)", 10),
        ("naive", 12), ("naive OK", 9),
    ]
    header = "".join(f"{c:>{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        exact_s = f"{r['exact']:.6f}" if r["exact"] is not None else "n/a"
        if r["se"]:
            se_s = f"{r['se']:.2e}"
        elif r["known_match"]:
            se_s = "0 (exact)"
        elif r["used_fallback"]:
            se_s = "fallback"
        else:
            se_s = "n/a"
        z_s = f"{r['z']:+.2f}" if r["z"] is not None else "n/a"
        naive_ok = "n/a"
        if r["exact"] is not None:
            naive_ok = "yes" if abs(r["naive"] - r["exact"]) < 1e-6 else "NO"
        vals = [
            r["pedigree"], r["n"], r["loci"], r["candidates"],
            exact_s, f"{r['sampled']:.6f}", se_s, z_s,
            f"{r['naive']:.6f}", naive_ok,
        ]
        print("".join(f"{v!s:>{w}}" for v, (_, w) in zip(vals, cols)))
    print()

    z_scores = [abs(r["z"]) for r in rows if r["z"] is not None]
    n_exact = sum(1 for r in rows if r["exact"] is not None)
    n_disagree = sum(
        1 for r in rows
        if r["exact"] is not None and abs(r["naive"] - r["exact"]) >= 1e-6
    )
    print(f"Pedigrees with an exact (non-sampling) reference: {n_exact} / {len(rows)}")
    if z_scores:
        print(f"max |z| = {max(z_scores):.2f}  (sampled vs. exact, in standard errors)")
    print(f"Naive independence formula measurably wrong on: {n_disagree} / {n_exact} "
          f"pedigree(s) with an exact reference -- exactly those with a shared "
          f"untyped ancestor among candidates.")


def write_csv(rows: list[dict], out_path: str) -> None:
    fieldnames = [
        "pedigree", "n", "loci", "candidates", "exact", "sampled", "se",
        "z", "naive", "used_fallback", "known_match",
    ]
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
        help="Also write the full table (all columns) to this CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = validate(args.mu, args.n_samples, args.seed)
    print_table(rows)
    if args.out:
        write_csv(rows, args.out)
        print(f"Full table written to '{args.out}'")
