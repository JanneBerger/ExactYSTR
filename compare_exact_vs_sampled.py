#!/usr/bin/env python3
"""
compare_exact_vs_sampled.py
============================
Validates the shipped i-MP sampling estimator (`compute_at_least_one_match`
in `ystr_pedigree.py`) against an exact, non-sampling reference, for the
single-copy A-E/G example pedigrees (A01, B01, C01, C02, D01, D02, E01,
E02, G01) and their two-copy (DYS385-like) analogues (A01_mc1, A01_mc2,
B01_mc, C01_mc, C02_mc, F01_mc). `compute_at_least_one_match[_exact]`
already treat `mu` as a per-copy rate and auto-detect each locus's copy
count, so the same comparison loop below handles single- and multi-copy
pedigrees alike.

`ystr_pedigree.py` ships the sampling method only (Section 3.2 of the
technical note): it is what scales to real pedigrees, since its cost is
independent of the number of untyped candidates. The exact method (full
inclusion-exclusion, equation 13) does not scale -- its cost is
O(2^|candidates|) -- so it is kept out of the core library and defined
locally in this validation script instead, reusing only
`_multicopy_inside_pass` (the shared inside-pass helper the public
`compute_match_probs_multicopy` already depends on).

Usage
-----
    python compare_exact_vs_sampled.py [--mu 0.1] [--n-samples 500000] \
        [--seed SEED] [--out results.csv]
"""

import argparse
import csv
import os
import time
from itertools import combinations

import numpy as np

from ystr_pedigree import (
    DEFAULT_MU,
    MAXALL,
    MINALL,
    Male,
    _infer_copies,
    _multicopy_inside_pass,
    _to_genotype,
    compute_at_least_one_match,
    load_pedigree_from_csv,
)

PEDIGREE_DIR = os.path.join(os.path.dirname(__file__), "Pedigrees")
AF_NAMES = [
    # single-copy
    "A01", "B01", "C01", "C02", "D01", "D02", "E01", "E02", "G01",
    # two-copy (DYS385-like) analogues
    "A01_mc1", "A01_mc2", "B01_mc", "C01_mc", "C02_mc", "F01_mc",
]


# ---------------------------------------------------------------------------
# Exact method (equation 13/14) -- validation-only, not part of the
# public API in ystr_pedigree.py
# ---------------------------------------------------------------------------

def _clamped_all_match_prob(males, locus, candidate_idx, P1, mu_all, mu2_all,
                             n_copies, minall, maxall):
    """P(every male in candidate_idx matches the suspect at locus), by
    clamping them to the suspect's observed genotype and re-peeling."""
    if not candidate_idx:
        return 1.0
    suspect_allele = males[0].alleles.get(locus, 0)
    clamped = [Male(m.male_id, m.generation, m.father_id, dict(m.alleles)) for m in males]
    for i in candidate_idx:
        clamped[i].alleles[locus] = suspect_allele
    ctx_s = _multicopy_inside_pass(
        clamped, locus, n_copies, mu_all, mu2_all, minall, maxall
    )
    if ctx_s is None:
        return 0.0
    return float(np.sum(ctx_s.beta[0])) / P1


def compute_at_least_one_match_exact(
    males, loci, mu=DEFAULT_MU, two_step_frac=0.0,
    minall=MINALL, maxall=MAXALL, max_candidates=20,
):
    """Exact i-MP via full inclusion-exclusion (equation 13). Cost is
    O(2^|candidates|) -- practical only for small candidate sets."""
    n = len(males)
    excluded = np.zeros(n, dtype=bool)
    loci_used = []
    per_locus_params = []

    for locus in loci:
        c = _infer_copies(males, locus)
        mu_all = 1.0 - (1.0 - mu) ** c
        lmu1 = mu_all * (1.0 - two_step_frac)
        lmu2 = mu_all * two_step_frac

        ctx = _multicopy_inside_pass(males, locus, c, lmu1, lmu2, minall, maxall)
        if ctx is None:
            continue
        P1 = float(np.sum(ctx.beta[0]))
        if P1 == 0.0:
            continue

        loci_used.append(locus)
        per_locus_params.append((locus, lmu1, lmu2, c, P1))
        for i, g in enumerate(ctx.base):
            if g is not None and tuple(sorted(g)) != tuple(sorted(ctx.suspect_g)):
                excluded[i] = True

    if not loci_used:
        return 0.0

    candidate_mask = ~excluded
    candidate_mask[0] = False
    candidates = [i for i in range(n) if candidate_mask[i]]

    known_match = any(
        all(_to_genotype(males[i].alleles.get(locus, 0)) is not None for locus in loci_used)
        for i in candidates
    )
    if known_match or not candidates:
        return 1.0 if known_match else 0.0

    if len(candidates) > max_candidates:
        raise ValueError(f"{len(candidates)} candidates exceeds max_candidates={max_candidates}")

    total = 0.0
    for k in range(1, len(candidates) + 1):
        for S in combinations(candidates, k):
            term = 1.0
            for locus, lmu1, lmu2, c, P1 in per_locus_params:
                term *= _clamped_all_match_prob(
                    males, locus, S, P1, lmu1, lmu2, c, minall, maxall
                )
            total += term if k % 2 == 1 else -term
    return total


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def run(mu: float, n_samples: int, seed: int) -> list[dict]:
    rows = []
    for name in AF_NAMES:
        path = os.path.join(PEDIGREE_DIR, f"{name}.csv")
        males, loci = load_pedigree_from_csv(path)

        t0 = time.perf_counter()
        sampled = compute_at_least_one_match(males, loci, mu=mu, n_samples=n_samples, seed=seed)
        t_sampled = time.perf_counter() - t0

        t0 = time.perf_counter()
        exact = compute_at_least_one_match_exact(males, loci, mu=mu)
        t_exact = time.perf_counter() - t0

        diff = sampled.i_mp - exact
        z = diff / sampled.se if sampled.se else (0.0 if diff == 0 else float("nan"))

        rows.append(dict(
            pedigree=f"{name}.csv", n=len(males), loci=len(loci),
            candidates=len(sampled.candidates),
            exact=exact, t_exact=t_exact,
            sampled=sampled.i_mp, se=sampled.se, t_sampled=t_sampled,
            diff=diff, z=z,
        ))
    return rows


def _fmt_t(t: float) -> str:
    return f"{t*1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def print_table(rows: list[dict]) -> None:
    cols = [
        ("pedigree", 12), ("n", 4), ("loci", 5), ("cand", 5),
        ("exact", 12), ("sampled", 12), ("diff", 12), ("z", 8),
        ("t_exact", 10), ("t_sampled", 10),
    ]
    header = "".join(f"{c:>{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [
            r["pedigree"], r["n"], r["loci"], r["candidates"],
            f"{r['exact']:.6f}", f"{r['sampled']:.6f}", f"{r['diff']:+.2e}",
            f"{r['z']:+.2f}", _fmt_t(r["t_exact"]), _fmt_t(r["t_sampled"]),
        ]
        print("".join(f"{v!s:>{w}}" for v, (_, w) in zip(vals, cols)))
    zs = [abs(r["z"]) for r in rows if r["z"] == r["z"]]  # drop NaN
    if zs:
        print(f"\nmax |z| = {max(zs):.2f} (sampled vs. exact, in standard errors)")


def write_csv(rows: list[dict], out_path: str) -> None:
    fieldnames = ["pedigree", "n", "loci", "candidates", "exact", "sampled",
                  "diff", "z", "t_exact", "t_sampled", "se"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fieldnames} for r in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--out", metavar="OUTPUT_CSV", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = run(args.mu, args.n_samples, args.seed)
    print_table(rows)
    if args.out:
        write_csv(rows, args.out)
        print(f"\nTable written to '{args.out}'")
