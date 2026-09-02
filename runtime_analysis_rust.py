"""
runtime_analysis_rust.py
========================
Builds and runs the Rust implementation of the Elston-Stewart algorithm,
then produces the identical 4-panel runtime figure as runtime_analysis.py.

Usage:
    python runtime_analysis_rust.py [--skip-build]

Output:
    runtime_analysis_rust.png / .pdf / .svg
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUST_DIR     = Path(__file__).parent / "ystr_rust"
BINARY       = RUST_DIR / "target" / "release" / "ystr_rust"
DATA_FILE    = Path(__file__).parent / "runtime_analysis_rust_data.json"
OUTPUT_BASE  = "runtime_analysis_rust"

TOPOLOGIES = {
    "Linear chain (1 son)":  1,
    "Wide (3 sons)":         3,
    "Very wide (5 sons)":    5,
}

COLORS = {
    "Linear chain (1 son)":  "#2166ac",
    "Wide (3 sons)":         "#d6604d",
    "Very wide (5 sons)":    "#1a9850",
}

MARKERS_STYLE = {
    "Linear chain (1 son)":  "o",
    "Wide (3 sons)":         "s",
    "Very wide (5 sons)":    "^",
}


# ---------------------------------------------------------------------------
# Build + run
# ---------------------------------------------------------------------------

def build_rust(skip_build: bool = False) -> None:
    if skip_build and BINARY.exists():
        print(f"Skipping build; using existing binary: {BINARY}")
        return
    print("Building Rust binary (release)…")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit("Rust build failed.")
    print("Build complete.\n")


def run_rust() -> dict:
    print(f"Running Rust experiments…  (output → {DATA_FILE})\n")
    result = subprocess.run(
        [str(BINARY)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit("Rust binary returned non-zero exit code.")

    data = json.loads(result.stdout)
    with open(DATA_FILE, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


# ---------------------------------------------------------------------------
# Plotting  (identical style to runtime_analysis.py)
# ---------------------------------------------------------------------------

def add_linear_fit(ax, xs, ys, color):
    coeffs = np.polyfit(xs, ys, 1)
    x_fit  = np.linspace(0, max(xs) * 1.05, 300)
    ax.plot(x_fit, np.polyval(coeffs, x_fit),
            "--", color=color, linewidth=1.2, alpha=0.6)


def plot_panel(ax, xs, results, stds, xlabel, panel_label,
               fixed_params="", y_zero_bottom=True):
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
    ax.text(-0.24, 1.0, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")
    if fixed_params:
        ax.text(0.97, 0.05, fixed_params, transform=ax.transAxes,
                fontsize=8, color="black", va="bottom", ha="right")


def make_figure(data: dict) -> None:
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       10,
        "axes.linewidth":  0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top":       True,
        "ytick.right":     True,
    })

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.5, top=0.95, bottom=0.12)

    plot_panel(axes[0, 0], data["ns"], data["res_n"], data["std_n"],
               xlabel="Pedigree size  $n_{tot}$",
               panel_label="a",
               fixed_params="$n_{unt} = 30$,  $n_{all} = 28$,  $n_{loc} = 1$")

    plot_panel(axes[0, 1], data["us"], data["res_u"], data["std_u"],
               xlabel="Number of untyped males  $n_{unt}$",
               panel_label="b",
               fixed_params="$n_{tot} = 200$,  $n_{all} = 28$,  $n_{loc} = 1$",
               y_zero_bottom=False)

    plot_panel(axes[1, 0], data["As"], data["res_A"], data["std_A"],
               xlabel="Number of alleles  $n_{all}$",
               panel_label="c",
               fixed_params="$n_{tot} = 200$,  $n_{unt} = 50$,  $n_{loc} = 1$")

    plot_panel(axes[1, 1], data["Ls"], data["res_L"], data["std_L"],
               xlabel="Number of loci  $n_{loc}$",
               panel_label="d",
               fixed_params="$n_{tot} = 50$,  $n_{unt} = 25$,  $n_{all} = 28$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=3,
               fontsize=9,
               frameon=False)

    fig.suptitle("Rust implementation  —  Elston-Stewart runtime", fontsize=11, y=0.99)

    for ext in ("png", "pdf", "svg"):
        out = f"{OUTPUT_BASE}.{ext}"
        plt.savefig(out, dpi=350, bbox_inches="tight")
        print(f"Figure saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    skip = "--skip-build" in sys.argv
    build_rust(skip_build=skip)
    data = run_rust()
    make_figure(data)
