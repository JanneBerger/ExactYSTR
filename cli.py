#!/usr/bin/env python3
"""
cli.py
======
Command-line interface for the exact Y-STR pedigree match probability
calculator based on the Elston-Stewart algorithm.

Usage
-----
    python cli.py pedigree.csv [--mu 0.1] [--out results.csv]

The pedigree CSV must contain columns:
    male_id, generation, father_id, allele_<LOCUS>, ...

See examples/example_pedigree.csv for the expected format.
All loci use a fixed allele range of [3, 30].
"""

import argparse
import csv
import sys

from ystr_pedigree import (
    DEFAULT_MU,
    MINALL,
    MAXALL,
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
        help="Mutation rate per meiosis.",
    )
    parser.add_argument(
        "--out",
        metavar="OUTPUT_CSV",
        default=None,
        help="Write results to this CSV file instead of stdout.",
    )
    return parser.parse_args()


def run(pedigree_path: str, mu: float, out_path: str | None) -> None:
    males, loci = load_pedigree_from_csv(pedigree_path)
    print(f"Loaded {len(males)} males, {len(loci)} loci from '{pedigree_path}'")
    print(f"Mutation rate: mu = {mu}")
    print(f"Allele range:  [{MINALL}, {MAXALL}]\n")

    rows: list[dict] = []

    for locus in loci:
        match_probs, average = compute_match_probs_fast(males, locus, mu)

        if not match_probs:
            print(f"[{locus}]  suspect untyped or no untyped males — skipped")
            continue

        print(f"[{locus}]  average MP = {average:.6f}")
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
        print()

    if out_path and rows:
        fieldnames = ["locus", "male_id", "generation", "match_prob"]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to '{out_path}'")


def main() -> None:
    args = parse_args()
    try:
        run(args.pedigree, args.mu, args.out)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
