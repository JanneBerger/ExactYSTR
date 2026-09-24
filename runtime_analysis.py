"""
runtime_analysis.py
====================
Empirical runtime analysis of the Elston-Stewart algorithm.

Measures runtime across four dimensions:
    (n)  pedigree size          — u = 30,   A = 28, L = 1
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

On top of these three lines, each panel also shows two overlay series
along the same x-axis, using a single fixed topology (wide3):
    multi-copy         — same computation, but for one c=2 locus
                          (compute_match_probs_multicopy) instead of one
                          single-copy locus
    at least one match — the i-MP sampling estimator
                          (compute_at_least_one_match) over the same
                          pedigree/loci, with a reduced sample count so
                          its runtime is on a comparable scale

The script imports directly from ystr_pedigree.py (GitHub implementation).
Place this file in the same directory as ystr_pedigree.py, or add its
location to PYTHONPATH.

Dependencies: numpy, matplotlib
Usage:
    python runtime_analysis.py
Output:
    runtime_analysis.png/.pdf/.svg  (publication-quality 4-panel figure,
    raster + vector formats)
"""

import sys
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Import from the GitHub implementation
# ---------------------------------------------------------------------------
from ystr_pedigree import (
    Male,
    compute_match_probs_fast as compute_match_probs,
    compute_match_probs_multicopy,
    compute_at_least_one_match,
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

N_SEEDS_EXTRA = 15             # seeds for the multi-copy overlay series
SUSPECT_MC    = 16             # allele carried by the suspect in multicopy experiments
A_MC          = MAXALL - MINALL + 1  # full allele range (no window narrowing)
MC_COPIES     = 2              # copy count used for the multi-copy overlay series
EXTRA_BRANCHING = 3            # fixed topology (wide3) used for the overlay series

N_SEEDS_IMP   = 5              # seeds for the dedicated i-MP panel
REPEATS_IMP   = 3              # fewer timing repeats (each call is much slower)
IMP_N_SAMPLES = 200_000        # the analysis default -- panel e is on its own
                                # scale now, so no need to shrink this for
                                # visual comparability with the other panels

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

MC_COLOR  = "#542788"   # purple — multi-copy overlay series
IMP_COLOR = "#e08214"   # orange — at-least-one-match overlay series


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


def generate_mixed_pedigree(
    n: int, n_untyped: int, branching: int,
    single_loci: list[str], multi_loci: list[str], n_copies: int,
    suspect_allele: int = SUSPECT_MC,
    seed: int = SEED,
) -> list[Male]:
    """Pedigree with some single-copy and some multi-copy loci.

    Typed males carry the (same) suspect allele at every ``single_loci``
    locus, and a tuple of ``n_copies`` copies of that allele at every
    ``multi_loci`` locus. The same set of untyped males applies at every
    locus.
    """
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
        alleles = {loc: (0 if i in untyped else suspect_allele) for loc in single_loci}
        alleles.update({
            loc: (0 if i in untyped else tuple([suspect_allele] * n_copies))
            for loc in multi_loci
        })
        males.append(Male(
            male_id    = i,
            generation = generation,
            father_id  = father_id,
            alleles    = alleles,
        ))

    males.sort(key=lambda m: (m.generation, m.male_id))
    return males


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure(fn, *args, repeats: int = REPEATS) -> float:
    """Return median CPU time in milliseconds over *repeats* calls.

    Uses time.process_time() (CPU time consumed by this process) rather
    than perf_counter() (wall-clock time), so measurements are immune to
    scheduling jitter from other processes on a shared/virtualized host.
    """
    # Warm-up: one call to populate caches before actual measurement
    fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.process_time()
        fn(*args)
        times.append(time.process_time() - t0)
    return float(np.median(times)) * 1000.0


def run_extra_series(xs: list, build_fn, n_seeds: int = N_SEEDS_EXTRA,
                      repeats: int = REPEATS):
    """Measure one overlay series (not split by topology) across xs,
    averaged over n_seeds random untyped-placements per data point.

    build_fn(x, seed) -> zero-arg callable to measure.
    """
    means, stds = [], []
    for x in xs:
        placement_times = [
            measure(build_fn(x, SEED + k), repeats=repeats)
            for k in range(n_seeds)
        ]
        means.append(float(np.mean(placement_times)))
        stds.append(float(np.std(placement_times)))
    return means, stds


# ---------------------------------------------------------------------------
# Experiments — main (per-topology) sweeps
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
# Experiments — multi-copy overlay series
#
# Each reuses the same swept parameter and fixed values as the matching
# per-topology experiment above, but with a single fixed topology
# (EXTRA_BRANCHING).
# ---------------------------------------------------------------------------

def overlay_n(ns):
    def build_mc(n, seed):
        males = generate_mixed_pedigree(n, 30, EXTRA_BRANCHING, [], [LOCUS], MC_COPIES, seed=seed)
        return lambda: compute_match_probs_multicopy(
            males, LOCUS, MC_COPIES, mu_all=MU, minall=MINALL, maxall=MAXALL
        )
    return run_extra_series(ns, build_mc)


def overlay_u(us, n=200):
    def build_mc(u, seed):
        males = generate_mixed_pedigree(n, u, EXTRA_BRANCHING, [], [LOCUS], MC_COPIES, seed=seed)
        return lambda: compute_match_probs_multicopy(
            males, LOCUS, MC_COPIES, mu_all=MU, minall=MINALL, maxall=MAXALL
        )
    return run_extra_series(us, build_mc)


def overlay_A(As, n=200, u=50):
    centre = (MINALL + MAXALL) // 2
    def build_mc(a_size, seed):
        mn, mx = centre - a_size // 2, centre - a_size // 2 + a_size - 1
        males = generate_mixed_pedigree(n, u, EXTRA_BRANCHING, [], [LOCUS], MC_COPIES,
                                         suspect_allele=centre, seed=seed)
        return lambda: compute_match_probs_multicopy(
            males, LOCUS, MC_COPIES, mu_all=MU, minall=mn, maxall=mx
        )
    return run_extra_series(As, build_mc)


def overlay_L(Ls, n=50, u=25):
    def build_mc(n_loci, seed):
        loci  = [f"L{i}" for i in range(n_loci)]
        males = generate_mixed_pedigree(n, u, EXTRA_BRANCHING, [], loci, MC_COPIES, seed=seed)
        def call():
            for loc in loci:
                compute_match_probs_multicopy(males, loc, MC_COPIES, mu_all=MU,
                                               minall=MINALL, maxall=MAXALL)
        return call
    return run_extra_series(Ls, build_mc)


# ---------------------------------------------------------------------------
# Experiment — dedicated at-least-one-match panel (panel e)
#
# Its own panel, on its own scale, using the analysis' real default
# n_samples (200,000) rather than a value shrunk for visual parity with
# panels a-d. x-axis reuses the untyped-male sweep (same n_tot=200 as
# panel b), since the untyped males are exactly the i-MP candidates.
# ---------------------------------------------------------------------------

def experiment_imp(us, n=200):
    print(f"[5/5] Sweep: i-MP runtime vs. number of untyped candidates  "
          f"(n_tot={n}, A=28, n_samples={IMP_N_SAMPLES:,}, {N_SEEDS_IMP} placements/point)")
    def build(u, seed):
        males = generate_pedigree(n, u, EXTRA_BRANCHING, [LOCUS], seed=seed)
        return lambda: compute_at_least_one_match(
            males, [LOCUS], mu=MU, n_samples=IMP_N_SAMPLES, seed=SEED
        )
    means, stds = run_extra_series(us, build, n_seeds=N_SEEDS_IMP, repeats=REPEATS_IMP)
    for u, m, s in zip(us, means, stds):
        print(f"  u={u:>4}   {m:.1f}±{s:.1f}ms")
    return means, stds


# ---------------------------------------------------------------------------
# Theoretical complexity
# ---------------------------------------------------------------------------

def print_complexity_summary():
    """Print the theoretical time-complexity of the algorithm."""
    print()
    print("=" * 62)
    print("  Theoretical complexity: O(n · c · A^c)  per locus")
    print("  n = number of males in pedigree")
    print("  c = number of copies  (1 = single-copy, 2 = DYS385, …)")
    print("  A = allele range (maxall - minall + 1); the full range is")
    print("      used for every copy, so the result is exact")
    print()
    print("  Inside pass  : O(c · A^c)  per male")
    print("  Outside pass : O(c · A^c)  per male   [same kernel, T symmetric]")
    print("  Over n males : O(n · c · A^c)  total per locus")
    print("  Over L loci  : O(L · n · c · A^c)  grand total")
    print()
    print("  Typical single-copy values (c=1, A=28):")
    print(f"    {'n':>6}  {'ops/locus':>12}")
    for n in (50, 100, 200, 500):
        print(f"    {n:>6}  {n * 28:>12,}")
    print()
    print(f"  Multi-copy (c=2, A={A_MC}):")
    print(f"    {'n':>6}  {'A^c':>6}  {'ops/locus':>12}")
    for n in (10, 50, 200):
        print(f"    {n:>6}  {A_MC**2:>6}  {n * 2 * A_MC**2:>12,}")
    print("=" * 62)
    print()


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
               fixed_params="", y_zero_bottom=True,
               linear_fit=True,
               mc_series=None):
    """Render one panel of the figure.

    results/stds: dict[topology_name -> list[float]], mean and standard
    deviation of runtime across the N_SEEDS random untyped-placements
    drawn per data point.

    linear_fit:
        Whether to overlay a per-topology dashed least-squares line fit.
    mc_series:
        Optional (means, stds) tuple for the multi-copy overlay series,
        plotted along the same xs.
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
        if linear_fit:
            add_linear_fit(ax, xs, ys, COLORS[name])

    if mc_series is not None:
        means, stds_mc = mc_series
        ax.errorbar(xs, means, yerr=stds_mc, marker="D", color=MC_COLOR,
                     linewidth=1.5, markersize=5, capsize=3, elinewidth=0.8,
                     linestyle="-.", label=f"multi-copy ($c={MC_COPIES}$)")

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


def plot_imp_panel(ax, xs, means, stds, panel_label):
    """Panel e: at-least-one-match runtime vs. number of untyped
    candidates, on its own scale with the analysis' real n_samples."""
    means_s = [m / 1000.0 for m in means]
    stds_s  = [s / 1000.0 for s in stds]
    ax.errorbar(xs, means_s, yerr=stds_s, marker="v", color=IMP_COLOR,
                 linewidth=1.5, markersize=5, capsize=3, elinewidth=0.8,
                 linestyle=":", label=f"at least one match ($n_{{samples}}$={IMP_N_SAMPLES:,})")

    ax.set_xlabel("Number of untyped candidates  $n_{unt}$", fontsize=10)
    ax.set_ylabel("Runtime (s)", fontsize=10)
    ax.set_xlim(left=0)
    ax.margins(y=0.15)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(labelsize=9)

    ax.text(-0.24, 1.0, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")
    ax.text(0.97, 0.05, "$n_{tot} = 200$,  $n_{all} = 28$,  $n_{loc} = 1$",
            transform=ax.transAxes, fontsize=8, color="black",
            va="bottom", ha="right")


def make_figure(ns, res_n, std_n, mc_n,
                us, res_u, std_u, mc_u,
                As, res_A, std_A, mc_A,
                Ls, res_L, std_L, mc_L,
                us_imp, means_imp, stds_imp):
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        10,
        "axes.linewidth":   0.8,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.top":        True,
        "ytick.right":      True,
    })

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.5, top=0.95, bottom=0.14)

    plot_panel(axes[0, 0], ns, res_n, std_n,
               xlabel="Pedigree size  $n_{tot}$",
               panel_label="a",
               fixed_params="$n_{unt} = 30$,  $n_{all} = 28$,  $n_{loc} = 1$",
               mc_series=mc_n)

    plot_panel(axes[0, 1], us, res_u, std_u,
               xlabel="Number of untyped males  $n_{unt}$",
               panel_label="b",
               fixed_params="$n_{tot} = 200$,  $n_{all} = 28$,  $n_{loc} = 1$",
               y_zero_bottom=False,
               mc_series=mc_u)

    plot_panel(axes[0, 2], As, res_A, std_A,
               xlabel="Number of alleles  $n_{all}$",
               panel_label="c",
               fixed_params="$n_{tot} = 200$,  $n_{unt} = 50$,  $n_{loc} = 1$",
               mc_series=mc_A)

    plot_panel(axes[1, 0], Ls, res_L, std_L,
               xlabel="Number of loci  $n_{loc}$",
               panel_label="d",
               fixed_params="$n_{tot} = 50$,  $n_{unt} = 25$,  $n_{all} = 28$",
               mc_series=mc_L)

    plot_imp_panel(axes[1, 1], us_imp, means_imp, stds_imp, panel_label="e")

    # Unused 6th slot in the 2x3 grid
    axes[1, 2].axis("off")

    # Shared legend: topology series + the multi-copy overlay (from panel
    # a) plus the at-least-one-match series (from panel e). Handles are
    # explicitly ordered (matplotlib fills a legend column-major) so
    # "at least one match" lands directly below "multi-copy" in the
    # same column (5 entries, ncol=2 -> column 0 gets 3, column 1 gets
    # the last 2).
    handles_a, labels_a = axes[0, 0].get_legend_handles_labels()
    handles_e, labels_e = axes[1, 1].get_legend_handles_labels()
    handle_map = dict(zip(labels_a, handles_a))
    handle_map.update(dict(zip(labels_e, handles_e)))

    order = [
        "Linear chain (1 son)", "Wide (3 sons)", "Very wide (5 sons)",
        f"multi-copy ($c={MC_COPIES}$)",
        f"at least one match ($n_{{samples}}$={IMP_N_SAMPLES:,})",
    ]
    ordered_labels  = [l for l in order if l in handle_map]
    ordered_handles = [handle_map[l] for l in ordered_labels]
    fig.legend(ordered_handles, ordered_labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=2,
               fontsize=9,
               frameon=False)

    base = "runtime_analysis"
    for ext in ("png", "pdf", "svg"):
        out = f"{base}.{ext}"
        plt.savefig(out, dpi=350, bbox_inches="tight")
        print(f"Figure saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_complexity_summary()

    # Sweep parameters
    ns = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
    us = [1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 199]
    As = [4, 6, 8, 10, 14, 18, 22, 26, 28]
    Ls = [1, 2, 3, 5, 8, 10, 15, 20]
    us_imp = [1, 10, 30, 50, 75, 100, 150, 199]  # fewer points: each is much slower

    res_n, std_n = experiment_n(ns)
    res_u, std_u = experiment_u(us)
    res_A, std_A = experiment_A(As)
    res_L, std_L = experiment_L(Ls)

    print("Overlay series: multi-copy, per panel")
    mc_n = overlay_n(ns)
    mc_u = overlay_u(us)
    mc_A = overlay_A(As)
    mc_L = overlay_L(Ls)

    means_imp, stds_imp = experiment_imp(us_imp)

    make_figure(ns, res_n, std_n, mc_n,
                us, res_u, std_u, mc_u,
                As, res_A, std_A, mc_A,
                Ls, res_L, std_L, mc_L,
                us_imp, means_imp, stds_imp)
