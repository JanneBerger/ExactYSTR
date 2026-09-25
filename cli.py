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

See Pedigrees/D01.csv for an example of the expected format.
All loci use a fixed allele range of [3, 30].

--mu is the overall mutation rate of a locus per meiosis (for multi-copy
loci: the all-copy rate mu_all, as usually reported for the locus as a
whole).  For multi-copy loci the per-copy rate mu = 1-(1-mu_all)^(1/c) is
derived automatically and shown in the console output.
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
    compute_at_least_one_match,
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
            "Overall mutation rate per locus and meiosis. For multi-copy "
            "loci this is the all-copy rate mu_all; the per-copy rate "
            "1-(1-mu_all)^(1/c) is derived automatically."
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
    parser.add_argument(
        "--at-least-one",
        action="store_true",
        help=(
            "Also compute i-MP, the probability that at least one untyped, "
            "non-excluded male anywhere in the pedigree matches the suspect "
            "at every locus simultaneously (exact sampling from the peeling "
            "messages; see the technical note, Section 3)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200_000,
        metavar="N",
        help="Number of posterior family draws used for --at-least-one.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --at-least-one (default: nondeterministic).",
    )
    return parser.parse_args()


def run(
    pedigree_path: str,
    mu: float,
    two_step_frac: float,
    out_path: str | None,
    at_least_one: bool = False,
    n_samples: int = 200_000,
    seed: int | None = None,
) -> None:
    males, loci = load_pedigree_from_csv(pedigree_path)
    print(f"Loaded {len(males)} males, {len(loci)} loci from '{pedigree_path}'")
    print(f"Mutation rate:  mu = {mu}  (per locus, all copies)")
    if two_step_frac:
        print(f"Two-step fraction: {two_step_frac}")
    print(f"Allele range:  [{MINALL}, {MAXALL}]\n")

    male_meta: dict[int, int] = {m.male_id: m.generation for m in males}
    combined: dict[int, float] = {}

    rows: list[dict] = []
    n_loci_used = 0
    total_elapsed = 0.0

    for locus in loci:
        c = _infer_copies(males, locus)

        # mu is the overall per-locus rate; the per-copy rate is derived
        # inside the multi-copy sweep (shown below for information)
        mu_pc = 1.0 - (1.0 - mu) ** (1.0 / c)
        lmu1 = mu * (1.0 - two_step_frac)
        lmu2 = mu * two_step_frac

        t0 = time.perf_counter()
        match_probs, average = compute_match_probs_fast(males, locus, lmu1, mu2=lmu2)
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        if not match_probs:
            print(f"[{locus}]  suspect untyped or no untyped males — skipped")
            continue

        n_loci_used += 1
        t_str = f"{elapsed*1000:.1f} ms" if elapsed < 1 else f"{elapsed:.3f} s"
        if c > 1:
            print(f"[{locus}]  {c} copies  →  per-copy mu = {mu_pc:.6g} (= 1-(1-{mu})^(1/{c}))  [{t_str}]")
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
        tot_str = f"{total_elapsed*1000:.1f} ms" if total_elapsed < 1 else f"{total_elapsed:.3f} s"
        print(f"=== Combined MP (product over {n_loci_used} loci, total {tot_str}) ===")
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

    if at_least_one:
        t0 = time.perf_counter()
        result = compute_at_least_one_match(
            males, loci, mu=mu, two_step_frac=two_step_frac,
            n_samples=n_samples, seed=seed,
        )
        elapsed = time.perf_counter() - t0
        t_str = f"{elapsed*1000:.1f} ms" if elapsed < 1 else f"{elapsed:.3f} s"

        print(f"=== At least one match, i-MP (over {len(result.loci_used)} loci, {t_str}) ===")
        if not result.candidates:
            print("    no untyped candidates")
            print(f"    i-MP = {result.i_mp:.6f}  (exact)")
        elif result.used_fallback:
            print(f"    i-MP = {result.i_mp:.6e}  "
                  f"(first-order inclusion-exclusion; below sampling resolution "
                  f"1/{n_samples})")
        else:
            print(f"    i-MP = {result.i_mp:.6f}  (se = {result.se:.2e}, "
                  f"n_samples = {n_samples})")
        print(f"    candidates: {len(result.candidates)}")
        print()


def main() -> None:
    args = parse_args()
    try:
        run(
            args.pedigree, args.mu, args.two_step_frac, args.out,
            at_least_one=args.at_least_one, n_samples=args.n_samples, seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
