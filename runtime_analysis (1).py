"""
runtime_analysis.py
====================
Empirical runtime analysis of the Elston-Stewart algorithm.

Measures runtime across four dimensions:
    (n)  pedigree size          — u = 49,   A = 28, L = 1
    (u)  number of untyped males — n = 200,  A = 28, L = 1
    (A)  allele range size       — n = 200,  u = 50, L = 1
    (L)  number of loci          — n = 50,   u = 25, A = 28

For each data point, the *placement* of the untyped males is re-drawn
N_SEEDS times (different random seeds), each placement measured (median
over REPEATS timing samples to filter out OS/GC noise), and the plotted
value is the mean +/- standard deviation across those N_SEEDS placements.
This captures how much runtime varies with *which* individuals happen to
be untyped, not just measurement noise on a single fixed layout.

Each experiment is run for three pedigree topologies:
    linear  — each male has exactly one son (chain)
    wide3   — each male has exactly three sons
    wide5   — each male has exactly five sons

The script imports directly from ystr_pedigree.py (GitHub implementation).
Place this file in the same directory as ystr_pedigree.py, or add its
location to PYTHONPATH.

Dependencies: numpy, matplotlib
Usage:
    python runtime_analysis.py
Output:
    runtime_analysis.png  (publication-quality 4-panel figure)
"""

import sys
import time
import random
import platform

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Import from the GitHub implementation
# ---------------------------------------------------------------------------
from ystr_pedigree import (
    Male,
    compute_match_probs_fast as compute_match_probs,
    MINALL,
    MAXALL,
    DEFAULT_MU,
)

sys.setrecursionlimit(500_000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU      = DEFAULT_MU   # 0.1
REPEATS = 5            # timing-noise samples (median) per untyped-placement
N_SEEDS = 20            # number of random untyped-placements per data point
LOCUS   = "L"          # generic locus name
SEED    = 22112000     # base seed; placement k uses SEED + k

TOPOLOGIES = {
    "Linear chain (1 son)":  1,
    "Wide (3 sons)":         3,
    "Very wide (5 sons)":    5,
}

COLORS = {
    "Linear chain (1 son)":  "#2166ac",   # blue
    "Wide (3 sons)":         "#d6604d",   # red
    "Very wide (5 sons)":    "#1a9850",   # green
}

MARKERS_STYLE = {
    "Linear chain (1 son)":  "o",
    "Wide (3 sons)":         "s",
    "Very wide (5 sons)":    "^",
}


# ---------------------------------------------------------------------------
# Pedigree generators
# ---------------------------------------------------------------------------

def generate_pedigree(n: int, n_untyped: int, branching: int,
                      loci: list[str], suspect_allele: int = 16,
                      seed: int = SEED,
                      fixed_untyped: set = None) -> list[Male]:
    """Generate a pedigree with a fixed branching factor.

    The pedigree is a complete b-ary tree of n males. The suspect (root,
    index 0) is always typed. A random subset of n_untyped other males
    is left untyped at all loci.

    Parameters
    ----------
    n:
        Total number of males.
    n_untyped:
        Number of untyped males (excluding root).
    branching:
        Number of sons per male (1 = linear chain).
    loci:
        List of locus names.
    suspect_allele:
        Allele carried by the suspect (and typed relatives).
    seed:
        Random seed for reproducibility.
    fixed_untyped:
        Optional set of male IDs to mark as untyped. If provided,
        n_untyped and seed are ignored for untyped selection.
    """
    if fixed_untyped is not None:
        untyped = fixed_untyped
    else:
        rng     = random.Random(seed)
        untyped = set(rng.sample(range(1, n), min(n_untyped, n - 1)))

    males = []
    for i in range(n):
        father_id = (i - 1) // branching if i > 0 else -1
        generation = 0
        j = i
        while j > 0:
            j = (j - 1) // branching
            generation += 1
        alleles = {
            loc: (0 if i in untyped else suspect_allele)
            for loc in loci
        }
        males.append(Male(
            male_id    = i,
            generation = generation,
            father_id  = father_id,
            alleles    = alleles,
        ))

    males.sort(key=lambda m: (m.generation, m.male_id))
    return males


# ---------------------------------------------------------------------------
# Measurement helper
# ---------------------------------------------------------------------------

def measure(fn, *args, repeats: int = REPEATS) -> float:
    """Return median runtime in milliseconds over *repeats* calls."""
    # Warm-up: one call to populate caches before actual measurement
    fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment(
    label: str, xs: list, build_args, n_seeds: int = N_SEEDS
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Run one sweep for all topologies, averaged over n_seeds random
    untyped-male placements per data point.

    Parameters
    ----------
    label:
        Display label for progress output.
    xs:
        Values of the swept parameter.
    build_args:
        Callable(x, branching, seed) -> (males, locus, mu, minall, maxall)
        or for multi-locus: returns a list of (males, locus, mu, minall, maxall)
        tuples; in that case we sum the runtimes.
    n_seeds:
        Number of random untyped-placements to draw per data point.

    Returns
    -------
    means, stds:
        Both dict[topology_name -> list[float]], one entry per x value.
        stds is the standard deviation of the per-placement runtime across
        the n_seeds different untyped-placements (structural variance),
        not the timing noise within a single placement.
    """
    means = {name: [] for name in TOPOLOGIES}
    stds  = {name: [] for name in TOPOLOGIES}

    for x in xs:
        for name, branching in TOPOLOGIES.items():
            placement_times = []
            for k in range(n_seeds):
                args_list = build_args(x, branching, SEED + k)
                # Support single or multi-locus calls
                if isinstance(args_list[0], tuple):
                    # multi-locus: sum runtime over loci
                    def call(al=args_list):
                        for a in al:
                            compute_match_probs(*a)
                    t = measure(call, repeats=REPEATS)
                else:
                    t = measure(compute_match_probs, *args_list, repeats=REPEATS)
                placement_times.append(t)
            means[name].append(float(np.mean(placement_times)))
            stds[name].append(float(np.std(placement_times)))

        vals = "  ".join(
            f"{name.split()[0][0]}={means[name][-1]:.1f}±{stds[name][-1]:.1f}ms"
            for name in TOPOLOGIES
        )
        print(f"  {label}={str(x):>5}   {vals}")

    return means, stds


def experiment_n(ns):
    print(f"[1/4] Sweep: pedigree size n  (u=30, A=28, L=1, {N_SEEDS} untyped-placements/point)")
    def build(n, branching, seed):
        males = generate_pedigree(n, 30, branching, [LOCUS], seed=seed)
        return males, LOCUS, MU, MINALL, MAXALL
    return run_experiment("n", ns, build)


def experiment_u(us, n=200):
    print(f"[2/4] Sweep: untyped males u  (n=200, A=28, L=1, {N_SEEDS} untyped-placements/point)")
    def build(u, branching, seed):
        males = generate_pedigree(n, u, branching, [LOCUS], seed=seed)
        return males, LOCUS, MU, MINALL, MAXALL
    return run_experiment("u", us, build)


def experiment_A(As, n=200, u=50):
    print(f"[3/4] Sweep: allele range A  (n=200, u=50, L=1, {N_SEEDS} untyped-placements/point)")
    centre = (MINALL + MAXALL) // 2
    def build(a_size, branching, seed):
        mn = centre - a_size // 2
        mx = mn + a_size - 1
        males = generate_pedigree(n, u, branching, [LOCUS], suspect_allele=centre, seed=seed)
        return males, LOCUS, MU, mn, mx
    return run_experiment("A", As, build)


def experiment_L(Ls, n=50, u=25):
    print(f"[4/4] Sweep: number of loci L  (n=50, u=25, A=28, {N_SEEDS} untyped-placements/point)")
    def build(n_loci, branching, seed):
        loci  = [f"L{i}" for i in range(n_loci)]
        males = generate_pedigree(n, u, branching, loci, seed=seed)
        return [(males, loc, MU, MINALL, MAXALL) for loc in loci]
    return run_experiment("L", Ls, build)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_linear_fit(ax, xs, ys, color):
    """Add a dashed linear fit line."""
    coeffs = np.polyfit(xs, ys, 1)
    x_fit  = np.linspace(0, max(xs) * 1.05, 300)
    ax.plot(x_fit, np.polyval(coeffs, x_fit),
            "--", color=color, linewidth=1.2, alpha=0.6)


def plot_panel(ax, xs, results, stds, xlabel, panel_label,
               show_legend=False, fixed_params="", y_zero_bottom=True):
    """Render one panel of the figure.

    results/stds: dict[topology_name -> list[float]], mean and standard
    deviation of runtime across the N_SEEDS random untyped-placements
    drawn per data point.
    """
    for name in TOPOLOGIES:
        ys  = results[name]
        err = stds[name]
        ax.errorbar(xs, ys, yerr=err,
                     marker=MARKERS_STYLE[name],
                     color=COLORS[name],
                     linewidth=1.5,
                     markersize=5,
                     capsize=3,
                     elinewidth=0.8,
                     label=name)
        add_linear_fit(ax, xs, ys, COLORS[name])

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Runtime (ms)", fontsize=10)
    ax.set_xlim(left=0)
    if y_zero_bottom:
        ax.set_ylim(bottom=0)
    else:
        ax.margins(y=0.15)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(labelsize=9)

    # Panel label (a, b, c, d), placed outside the axes to the upper-left
    ax.text(-0.24, 1.0, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            va="top", ha="right")

    # Fixed parameter annotation bottom-right
    if fixed_params:
        ax.text(0.97, 0.05, fixed_params, transform=ax.transAxes,
                fontsize=8, color="black", va="bottom", ha="right")

    pass  # legend is drawn once outside all panels in make_figure


def make_figure(ns, res_n, std_n, us, res_u, std_u, As, res_A, std_A, Ls, res_L, std_L):
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        10,
        "axes.linewidth":   0.8,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.top":        True,
        "ytick.right":      True,
    })

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.5, top=0.95, bottom=0.12)

    plot_panel(axes[0, 0], ns, res_n, std_n,
               xlabel="Pedigree size  $n_{tot}$",
               panel_label="a",
               show_legend=True,
               fixed_params="$n_{unt} = 30$,  $n_{all} = 28$,  $n_{loc} = 1$")

    plot_panel(axes[0, 1], us, res_u, std_u,
               xlabel="Number of untyped males  $n_{unt}$",
               panel_label="b",
               fixed_params="$n_{tot} = 200$,  $n_{all} = 28$,  $n_{loc} = 1$",
               y_zero_bottom=False)

    plot_panel(axes[1, 0], As, res_A, std_A,
               xlabel="Number of alleles  $n_{all}$",
               panel_label="c",
               fixed_params="$n_{tot} = 200$,  $n_{unt} = 50$,  $n_{loc} = 1$")

    plot_panel(axes[1, 1], Ls, res_L, std_L,
               xlabel="Number of loci  $n_{loc}$",
               panel_label="d",
               fixed_params="$n_{tot} = 50$,  $n_{unt} = 25$,  $n_{all} = 28$")

    # Single shared legend centered below all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=3,
               fontsize=9,
               frameon=False)

    fig.text(0.5, -0.02,
             f"Mean ± std over {N_SEEDS} untyped-placements  ·  "
             f"{REPEATS} timing replicates each  ·  "
             f"Python {platform.python_version()}  ·  "
             f"{platform.processor() or platform.machine()}  ·  "
             f"$\\mu$ = {MU}",
             ha="center", fontsize=7.5, color="gray")

    out = "runtime_analysis.png"
    plt.savefig(out, dpi=350, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sweep parameters
    ns = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
    us = [1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 199]
    As = [4, 6, 8, 10, 14, 18, 22, 26, 28]
    Ls = [1, 2, 3, 5, 8, 10, 15, 20]

    res_n, std_n = experiment_n(ns)
    res_u, std_u = experiment_u(us)
    res_A, std_A = experiment_A(As)
    res_L, std_L = experiment_L(Ls)

    make_figure(ns, res_n, std_n, us, res_u, std_u, As, res_A, std_A, Ls, res_L, std_L)
