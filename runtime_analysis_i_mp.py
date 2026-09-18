#!/usr/bin/env python3
"""
runtime_analysis_i_mp.py
=========================
Empirical runtime scaling of exact vs. sampled i-MP as the number of
candidates grows, from 1 up to and including ``--max-k`` (default 20).

Pedigree shape: a typed suspect with k independent, untyped sons at a
single locus (a "star" pedigree). Candidates are conditionally
independent given the suspect, so this isolates the one thing that
actually drives the cost difference between the two methods -- the
*candidate count* -- from pedigree depth, marker count, or dependency
structure.

  exact
      ``compute_at_least_one_match_exact`` -- full inclusion-exclusion
      (equation 13), cost ``O(2^k)``.
  sampled
      ``compute_at_least_one_match`` -- exact-sampling estimator
      (Section 3.2), cost dominated by ``n_samples`` and roughly
      independent of k.

This is the direct, measured version of the extrapolation discussed
alongside it: exact roughly doubles in time for every extra candidate,
while sampled stays flat. Because exact's cost is exponential, this
script's own running time is dominated by its last couple of k values
(k=20 alone took ~6.4 minutes in an earlier one-off measurement) --
expect the full 1..20 sweep to take on the order of 10-15 minutes.

Usage
-----
    python runtime_analysis_i_mp.py [--max-k 20] [--mu 0.1] \
        [--n-samples 200000] [--seed SEED] [--out results.csv] [--plot figure.png]
"""

import argparse
import csv
import time

from ystr_pedigree import (
    DEFAULT_MU,
    Male,
    compute_at_least_one_match,
    compute_at_least_one_match_exact,
)


def star_pedigree(k: int) -> list[Male]:
    """Typed suspect (allele 13) with k untyped, independent sons."""
    males = [Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13})]
    for i in range(k):
        males.append(Male(male_id=i + 2, generation=1, father_id=1, alleles={"M1": 0}))
    return males


def run(max_k: int, mu: float, n_samples: int, seed: int) -> list[dict]:
    rows = []
    for k in range(1, max_k + 1):
        males = star_pedigree(k)

        t0 = time.perf_counter()
        exact = compute_at_least_one_match_exact(males, ["M1"], mu=mu, max_candidates=max_k)
        t_exact = time.perf_counter() - t0

        t0 = time.perf_counter()
        sampled = compute_at_least_one_match(
            males, ["M1"], mu=mu, n_samples=n_samples, seed=seed
        )
        t_sampled = time.perf_counter() - t0

        row = dict(k=k, exact=exact, t_exact=t_exact,
                   sampled=sampled.i_mp, t_sampled=t_sampled)
        rows.append(row)
        ratio = t_exact / t_sampled if t_sampled > 0 else float("inf")
        print(f"k={k:3d}  exact={exact:.6f} ({t_exact:8.3f}s)   "
              f"sampled={sampled.i_mp:.6f} ({t_sampled:6.3f}s)   "
              f"exact/sampled time ratio = {ratio:8.2f}x")
    return rows


def write_csv(rows: list[dict], out_path: str) -> None:
    fieldnames = ["k", "exact", "t_exact", "sampled", "t_sampled"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_COLOR_EXACT   = "#2a78d6"  # blue  (project default palette, slot 1)
_COLOR_SAMPLED = "#eb6834"  # orange (slot 2)
_COLOR_INK     = "#0b0b0b"
_COLOR_MUTED   = "#898781"
_COLOR_GRID    = "#e1e0d9"
_COLOR_SURFACE = "#fcfcfb"


def plot_scaling(rows: list[dict], out_path: str) -> None:
    """Runtime vs. candidate count, log-scale y, both methods as lines."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    ks = [r["k"] for r in rows]
    t_exact = [r["t_exact"] for r in rows]
    t_sampled = [r["t_sampled"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    ax.plot(ks, t_exact, marker="o", markersize=5, linewidth=2,
            color=_COLOR_EXACT, label="exact (inclusion-exclusion, O(2^k))", zorder=3)
    ax.plot(ks, t_sampled, marker="o", markersize=5, linewidth=2,
            color=_COLOR_SAMPLED, label="sampled (exact-sampling estimator)", zorder=3)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v*1000:g} ms" if v < 1 else f"{v:g} s"
    ))
    ax.set_xlabel("number of candidates (k)", color=_COLOR_INK)
    ax.set_ylabel("wall-clock time (log scale)", color=_COLOR_INK)
    ax.set_xticks(ks)
    ax.tick_params(axis="both", colors=_COLOR_MUTED)
    ax.tick_params(axis="x", colors=_COLOR_INK)

    ax.grid(axis="y", which="major", color=_COLOR_GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color=_COLOR_GRID, linewidth=0.4, zorder=0)
    ax.grid(axis="x", color=_COLOR_GRID, linewidth=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_COLOR_MUTED)

    ax.set_title(
        "i-MP runtime vs. candidate count: exact vs. sampled",
        color=_COLOR_INK, fontsize=13, loc="left", pad=12,
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=_COLOR_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=_COLOR_SURFACE)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--out", metavar="OUTPUT_CSV", default=None)
    parser.add_argument("--plot", metavar="FIGURE_PNG", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = run(args.max_k, args.mu, args.n_samples, args.seed)
    if args.out:
        write_csv(rows, args.out)
        print(f"\nTable written to '{args.out}'")
    if args.plot:
        plot_scaling(rows, args.plot)
        print(f"Figure written to '{args.plot}'")
