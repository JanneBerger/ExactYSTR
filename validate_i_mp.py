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
    python validate_i_mp.py [--mu MU] [--n-samples N] [--seed SEED] \
        [--out results.csv] [--plot figure.png]
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


# Fixed categorical order (slots 1-2 of the project's default palette):
# validated colorblind-safe as an adjacent pair, never reassigned by rank.
_COLOR_EXACT   = "#2a78d6"  # blue
_COLOR_SAMPLED = "#eb6834"  # orange
_COLOR_INK     = "#0b0b0b"
_COLOR_MUTED   = "#898781"
_COLOR_GRID    = "#e1e0d9"
_COLOR_SURFACE = "#fcfcfb"


_SHORT_NAME = {
    "Large_Pedigree": "Large",
    "Large_Pedigree_mut": "Large_mut",
}


def plot_runtime(rows: list[dict], out_path: str) -> None:
    """Grouped bar chart of exact vs. sampled wall-clock time per pedigree
    (log scale -- runtimes span ms to tens of seconds), sorted by
    candidate count so complexity increases left to right."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rows = sorted(rows, key=lambda r: r["candidates"])
    n = len(rows)
    names = [r["pedigree"].removesuffix(".csv") for r in rows]
    names = [_SHORT_NAME.get(nm, nm) for nm in names]
    labels = [f"{nm}\n({r['candidates']} cand.)" for nm, r in zip(names, rows)]
    t_exact   = [r["t_exact"] for r in rows]
    t_sampled = [r["t_sampled"] for r in rows]

    x = list(range(n))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(max(12, n * 1.05), 6), facecolor=_COLOR_SURFACE
    )
    ax.set_facecolor(_COLOR_SURFACE)

    bars_sampled = ax.bar(
        [i + width / 2 for i in x], t_sampled, width,
        label="sampled (exact-sampling estimator)", color=_COLOR_SAMPLED,
        zorder=3,
    )
    bars_exact = ax.bar(
        [i - width / 2 for i in x], [t if t is not None else 0 for t in t_exact],
        width, label="exact (inclusion-exclusion, no sampling)", color=_COLOR_EXACT,
        zorder=3,
    )

    def _label_bars(bars, values, skip_missing=False):
        for bar, v in zip(bars, values):
            if v is None:
                if not skip_missing:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, 1.6e-3, "n/a",
                        ha="center", va="bottom", fontsize=8, color=_COLOR_MUTED,
                    )
                continue
            label = f"{v*1000:.1f} ms" if v < 1 else f"{v:.1f} s"
            ax.text(
                bar.get_x() + bar.get_width() / 2, v * 1.2, label,
                ha="center", va="bottom", fontsize=8, color=_COLOR_INK,
            )

    _label_bars(bars_exact, t_exact)
    _label_bars(bars_sampled, t_sampled)

    ax.set_yscale("log")
    ax.set_ylim(1e-4, max(t for t in t_sampled if t) * 8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v*1000:g} ms" if v < 1 else f"{v:g} s"
    ))
    ax.set_ylabel("wall-clock time (log scale)", color=_COLOR_INK)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=_COLOR_INK, fontsize=9)
    ax.tick_params(axis="y", colors=_COLOR_MUTED)
    ax.tick_params(axis="x", length=0)

    ax.grid(axis="y", which="major", color=_COLOR_GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color=_COLOR_GRID, linewidth=0.4, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_COLOR_MUTED)

    ax.set_title(
        "Exact vs. sampled i-MP: wall-clock time per pedigree",
        color=_COLOR_INK, fontsize=13, loc="left", pad=12,
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=_COLOR_INK)
    fig.text(
        0.01, 0.01,
        "n/a: candidate count exceeds the exact method's practical limit "
        f"({EXACT_MAX_CANDIDATES}; cost is O(2^|candidates|))",
        fontsize=8, color=_COLOR_MUTED,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=_COLOR_SURFACE)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--out", metavar="OUTPUT_CSV", default=None,
        help="Also write the table as CSV.",
    )
    parser.add_argument(
        "--plot", metavar="FIGURE_PNG", default=None,
        help="Also save a runtime comparison figure (requires matplotlib).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = validate(args.mu, args.n_samples, args.seed)
    print_table(rows)
    if args.out:
        write_csv(rows, args.out)
        print(f"\nTable written to '{args.out}'")
    if args.plot:
        plot_runtime(rows, args.plot)
        print(f"Figure written to '{args.plot}'")
