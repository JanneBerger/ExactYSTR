#!/usr/bin/env python3
"""
cli.py
======
Command-line interface for the exact Y-STR pedigree match probability
calculator based on the Elston-Stewart algorithm.

Usage
-----
    python cli.py pedigree.csv [--mu 0.1] [--two-step-frac 0.0] [--out results.csv]

The pedigree CSV must contain columns:
    male_id, generation, father_id, allele_<LOCUS>, ...

See examples/example_pedigree.csv for the expected format.
All loci use a fixed allele range of [3, 30].

--mu is always the per-copy mutation rate (mu_pc).  For multi-copy loci
the per-locus rate mu_all = 1-(1-mu)^c is derived automatically and shown
in the console output.
"""

import argparse
import csv
import sys
import time

from ystr_pedigree import (
    DEFAULT_MU,
    MINALL,
    MAXALL,
    _infer_copies,
    compute_match_probs_fast,
    load_pedigree_from_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact Y-STR pedigree match probabilities (Elston-Stewart algorithm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pedigree",
        metavar="PEDIGREE_CSV",
        help="Path to the input pedigree CSV file.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=DEFAULT_MU,
        help=(
            "Per-copy mutation rate (mu_pc). For multi-copy loci the "
            "per-locus rate mu_all = 1-(1-mu)^c is derived automatically. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--two-step-frac",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Fraction of mutations that are two-step (0–1, default: 0).",
    )
    parser.add_argument(
        "--out",
        metavar="OUTPUT_CSV",
        default=None,
        help="Write results to this CSV file instead of stdout.",
    )
    return parser.parse_args()


def run(pedigree_path: str, mu: float, two_step_frac: float, out_path: str | None) -> None:
    males, loci = load_pedigree_from_csv(pedigree_path)
    print(f"Loaded {len(males)} males, {len(loci)} loci from '{pedigree_path}'")
    print(f"Mutation rate:  mu = {mu}  (per copy)")
    if two_step_frac:
        print(f"Two-step fraction: {two_step_frac}")
    print(f"Allele range:  [{MINALL}, {MAXALL}]\n")

    male_meta: dict[int, int] = {m.male_id: m.generation for m in males}
    combined: dict[int, float] = {}

    rows: list[dict] = []
    n_loci_used = 0

    for locus in loci:
        c = _infer_copies(males, locus)

        # mu is per-copy; derive per-locus all-copy rate for this locus
        mu_all = 1.0 - (1.0 - mu) ** c
        lmu1 = mu_all * (1.0 - two_step_frac)
        lmu2 = mu_all * two_step_frac

        t0 = time.perf_counter()
        match_probs, average = compute_match_probs_fast(males, locus, lmu1, mu2=lmu2)
        elapsed = time.perf_counter() - t0

        if not match_probs:
            print(f"[{locus}]  suspect untyped or no untyped males — skipped")
            continue

        n_loci_used += 1
        t_str = f"{elapsed*1000:.1f} ms" if elapsed < 1 else f"{elapsed:.3f} s"
        if c > 1:
            print(f"[{locus}]  {c} copies  →  mu_all = {mu_all:.6g} (= 1-(1-{mu})^{c})  [{t_str}]")
            print(f"    average MP = {average:.6f}")
        else:
            print(f"[{locus}]  average MP = {average:.6f}  [{t_str}]")
        for i, prob in match_probs.items():
            male_id = males[i].male_id
            gen     = males[i].generation
            print(f"    male_id={male_id}  gen={gen}  MP={prob:.6f}")
            rows.append({
                "locus":      locus,
                "male_id":    male_id,
                "generation": gen,
                "match_prob": prob,
            })
            combined[male_id] = combined.get(male_id, 1.0) * prob
        print()

    if n_loci_used > 1 and combined:
        print(f"=== Combined MP (product over {n_loci_used} loci) ===")
        for male_id, mp in sorted(combined.items(), key=lambda x: male_meta[x[0]]):
            gen = male_meta[male_id]
            print(f"    male_id={male_id}  gen={gen}  combined_MP={mp:.6e}")
        print()

    if out_path and rows:
        fieldnames = ["locus", "male_id", "generation", "match_prob"]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            if n_loci_used > 1:
                writer.writerows([
                    {
                        "locus":      "COMBINED",
                        "male_id":    male_id,
                        "generation": male_meta[male_id],
                        "match_prob": mp,
                    }
                    for male_id, mp in sorted(combined.items())
                ])
        print(f"Results written to '{out_path}'")


def main() -> None:
    args = parse_args()
    try:
        run(args.pedigree, args.mu, args.two_step_frac, args.out)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
