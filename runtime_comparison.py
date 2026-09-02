"""
runtime_comparison.py
=====================
Runs (or loads cached) Python and Rust runtime experiments and plots both
implementations side-by-side in the same 4-panel figure.

Usage:
    python runtime_comparison.py              # run everything fresh
    python runtime_comparison.py --skip-rust  # reuse existing Rust data
    python runtime_comparison.py --skip-py    # reuse existing Python data
    python runtime_comparison.py --skip-rust --skip-py  # plot from cache only

Output:
    runtime_comparison.{png,pdf,svg}
"""

import json
import subprocess
import sys
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
RUST_DIR     = SCRIPT_DIR / "ystr_rust"
RUST_BINARY  = RUST_DIR / "target" / "release" / "ystr_analysis"
PY_CACHE     = SCRIPT_DIR / "runtime_analysis_python_data.json"
RUST_CACHE   = SCRIPT_DIR / "runtime_analysis_rust_data.json"
OUTPUT_BASE  = "runtime_comparison"

# ---------------------------------------------------------------------------
# Shared experiment parameters
# ---------------------------------------------------------------------------
NS = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
US = [1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 199]
AS = [4, 6, 8, 10, 14, 18, 22, 26, 28]
LS = [1, 2, 3, 5, 8, 10, 15, 20]

TOPOLOGIES = {
    "Linear chain (1 son)": 1,
    "Wide (3 sons)":        3,
    "Very wide (5 sons)":   5,
}
COLORS = {
    "Linear chain (1 son)": "#2166ac",
    "Wide (3 sons)":        "#d6604d",
    "Very wide (5 sons)":   "#1a9850",
}
MARKERS = {
    "Linear chain (1 son)": "o",
    "Wide (3 sons)":        "s",
    "Very wide (5 sons)":   "^",
}

MU       = 0.1
MINALL   = 3
MAXALL   = 30
REPEATS  = 5
N_SEEDS  = 20
SEED     = 22112000
LOCUS    = "L"

# ---------------------------------------------------------------------------
# Python experiments
# ---------------------------------------------------------------------------

def _py_generate(n, n_untyped, branching, loci, seed):
    from ystr_pedigree import Male
    rng     = random.Random(seed)
    untyped = set(rng.sample(range(1, n), min(n_untyped, n - 1)))
    males   = []
    for i in range(n):
        father_id  = (i - 1) // branching if i > 0 else -1
        generation = 0
        j = i
        while j > 0:
            j = (j - 1) // branching
            generation += 1
        alleles = {loc: (0 if i in untyped else 16) for loc in loci}
        males.append(Male(male_id=i, generation=generation,
                          father_id=father_id, alleles=alleles))
    males.sort(key=lambda m: (m.generation, m.male_id))
    return males


def _py_measure(fn, *args):
    fn(*args)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def run_python_experiments():
    from ystr_pedigree import compute_match_probs_fast as cmp

    def sweep(label, xs, build):
        means = {n: [] for n in TOPOLOGIES}
        stds  = {n: [] for n in TOPOLOGIES}
        for x in xs:
            for name, br in TOPOLOGIES.items():
                pts = []
                for k in range(N_SEEDS):
                    items = build(x, br, SEED + k)
                    if isinstance(items[0], tuple):
                        def call(al=items):
                            for a in al: cmp(*a)
                        t = _py_measure(call)
                    else:
                        t = _py_measure(cmp, *items)
                    pts.append(t)
                means[name].append(float(np.mean(pts)))
                stds[name].append(float(np.std(pts)))
            parts = "  ".join(
                f"{n[0]}={means[n][-1]:.2f}±{stds[n][-1]:.2f}ms"
                for n in TOPOLOGIES)
            print(f"  {label}={str(x):>5}   {parts}")
        return means, stds

    def build_n(n, br, seed):
        m = _py_generate(n, 30, br, [LOCUS], seed)
        return m, LOCUS, MU, MINALL, MAXALL

    def build_u(u, br, seed):
        m = _py_generate(200, u, br, [LOCUS], seed)
        return m, LOCUS, MU, MINALL, MAXALL

    def build_a(a, br, seed):
        centre = (MINALL + MAXALL) // 2
        mn = centre - a // 2
        mx = mn + a - 1
        m = _py_generate(200, 50, br, [LOCUS], seed)
        return m, LOCUS, MU, mn, mx

    def build_l(nl, br, seed):
        loci = [f"L{i}" for i in range(nl)]
        m = _py_generate(50, 25, br, loci, seed)
        return [(m, loc, MU, MINALL, MAXALL) for loc in loci]

    print(f"[Python 1/4] Sweep n  (u=30, A=28, L=1, {N_SEEDS} seeds/point)")
    res_n, std_n = sweep("n", NS, build_n)
    print(f"[Python 2/4] Sweep u  (n=200, A=28, L=1, {N_SEEDS} seeds/point)")
    res_u, std_u = sweep("u", US, build_u)
    print(f"[Python 3/4] Sweep A  (n=200, u=50, L=1, {N_SEEDS} seeds/point)")
    res_a, std_a = sweep("A", AS, build_a)
    print(f"[Python 4/4] Sweep L  (n=50, u=25, A=28, {N_SEEDS} seeds/point)")
    res_l, std_l = sweep("L", LS, build_l)

    data = dict(
        ns=NS, res_n=res_n, std_n=std_n,
        us=US, res_u=res_u, std_u=std_u,
        As=AS, res_A=res_a, std_A=std_a,
        Ls=LS, res_L=res_l, std_L=std_l,
    )
    with open(PY_CACHE, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nPython data cached → {PY_CACHE}\n")
    return data


def load_python_data(skip: bool) -> dict:
    if skip and PY_CACHE.exists():
        print(f"Loading cached Python data from {PY_CACHE}")
        with open(PY_CACHE) as fh:
            return json.load(fh)
    return run_python_experiments()

# ---------------------------------------------------------------------------
# Rust experiments
# ---------------------------------------------------------------------------

def build_rust(skip: bool) -> None:
    if skip and RUST_BINARY.exists():
        print(f"Skipping Rust build; using {RUST_BINARY}")
        return
    print("Building Rust binaries (release)…")
    r = subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR)
    if r.returncode != 0:
        sys.exit("Rust build failed.")
    print("Build complete.\n")


def load_rust_data(skip: bool) -> dict:
    if skip and RUST_CACHE.exists():
        print(f"Loading cached Rust data from {RUST_CACHE}")
        with open(RUST_CACHE) as fh:
            return json.load(fh)
    print(f"Running Rust experiments…  (output → {RUST_CACHE})\n")
    r = subprocess.run([str(RUST_BINARY)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit("Rust binary returned non-zero exit code.")
    data = json.loads(r.stdout)
    with open(RUST_CACHE, "w") as fh:
        json.dump(data, fh, indent=2)
    return data

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_fit(ax, xs, ys, color, ls):
    coeffs = np.polyfit(xs, ys, 1)
    xf = np.linspace(0, max(xs) * 1.05, 300)
    ax.plot(xf, np.polyval(coeffs, xf), color=color,
            linewidth=1.0, alpha=0.5, linestyle=ls)


def plot_panel(ax, xs, py_res, py_std, rs_res, rs_std,
               xlabel, panel_label, fixed_params="", y_zero_bottom=True):
    for name in TOPOLOGIES:
        col = COLORS[name]
        mk  = MARKERS[name]

        # Python — dashed, open marker
        ax.errorbar(xs, py_res[name], yerr=py_std[name],
                    marker=mk, color=col, linewidth=1.2, markersize=5,
                    capsize=3, elinewidth=0.7,
                    linestyle="--", fillstyle="none")
        _add_fit(ax, xs, py_res[name], col, "--")

        # Rust — solid, filled marker
        ax.errorbar(xs, rs_res[name], yerr=rs_std[name],
                    marker=mk, color=col, linewidth=1.5, markersize=5,
                    capsize=3, elinewidth=0.8,
                    linestyle="-")
        _add_fit(ax, xs, rs_res[name], col, "-")

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
    ax.text(-0.24, 1.0, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")
    if fixed_params:
        ax.text(0.97, 0.05, fixed_params, transform=ax.transAxes,
                fontsize=8, color="black", va="bottom", ha="right")


def make_figure(py: dict, rs: dict) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.52, top=0.93, bottom=0.16)

    plot_panel(axes[0, 0], py["ns"], py["res_n"], py["std_n"],
               rs["res_n"], rs["std_n"],
               xlabel="Pedigree size  $n_{tot}$", panel_label="a",
               fixed_params="$n_{unt}=30$,  $n_{all}=28$,  $n_{loc}=1$")

    plot_panel(axes[0, 1], py["us"], py["res_u"], py["std_u"],
               rs["res_u"], rs["std_u"],
               xlabel="Untyped males  $n_{unt}$", panel_label="b",
               fixed_params="$n_{tot}=200$,  $n_{all}=28$,  $n_{loc}=1$",
               y_zero_bottom=False)

    plot_panel(axes[1, 0], py["As"], py["res_A"], py["std_A"],
               rs["res_A"], rs["std_A"],
               xlabel="Number of alleles  $n_{all}$", panel_label="c",
               fixed_params="$n_{tot}=200$,  $n_{unt}=50$,  $n_{loc}=1$")

    plot_panel(axes[1, 1], py["Ls"], py["res_L"], py["std_L"],
               rs["res_L"], rs["std_L"],
               xlabel="Number of loci  $n_{loc}$", panel_label="d",
               fixed_params="$n_{tot}=50$,  $n_{unt}=25$,  $n_{all}=28$")

    # Legend: topology colours
    topo_handles = [
        mlines.Line2D([], [], color=COLORS[n], marker=MARKERS[n],
                      markersize=6, linewidth=1.5, label=n)
        for n in TOPOLOGIES
    ]
    # Legend: implementation line styles
    impl_handles = [
        mlines.Line2D([], [], color="black", linewidth=1.5,
                      linestyle="-",  label="Rust"),
        mlines.Line2D([], [], color="black", linewidth=1.2,
                      linestyle="--", label="Python",
                      marker="o", markersize=5, fillstyle="none"),
    ]

    leg1 = fig.legend(handles=topo_handles, loc="lower center",
                      bbox_to_anchor=(0.30, 0.01), ncol=1,
                      fontsize=8.5, frameon=False)
    fig.legend(handles=impl_handles, loc="lower center",
               bbox_to_anchor=(0.75, 0.01), ncol=1,
               fontsize=8.5, frameon=False)
    fig.add_artist(leg1)

    fig.suptitle("Elston-Stewart runtime  —  Python vs. Rust",
                 fontsize=11, y=0.975)

    for ext in ("png", "pdf", "svg"):
        out = f"{OUTPUT_BASE}.{ext}"
        plt.savefig(out, dpi=350, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    skip_rust = "--skip-rust" in sys.argv
    skip_py   = "--skip-py"   in sys.argv

    if not skip_rust:
        build_rust(skip=False)

    py_data = load_python_data(skip=skip_py)
    rs_data = load_rust_data(skip=skip_rust)

    make_figure(py_data, rs_data)
