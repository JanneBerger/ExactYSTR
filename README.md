# Exact Y-STR Match Probabilities

Exact computation of Y-chromosomal STR (Y-STR) match probabilities within a
pedigree, using the Elston-Stewart algorithm.

Given a pedigree with a typed suspect and one or more relatives who are
untyped (missing) at a locus, this tool computes, for each untyped male, the
probability that he carries the suspect's allele - exactly, accounting for mutation between generations.

## Background

Y-STR haplotypes are passed down largely unchanged from father to son, so
close paternal relatives of a suspect can share, or nearly share, a Y-STR
profile. When a relative in the pedigree is untyped at a locus, this
package computes the exact conditional match probability

```
MP_i = P(observed data | male i carries the suspect's allele) / P(observed data)
```

for every untyped male `i`, via backward (and forward) induction over the
pedigree tree rather than by enumerating or sampling genotype
configurations.

It also computes **i-MP**, the probability that *at least one* untyped,
non-excluded male anywhere in the pedigree matches the suspect at every
locus simultaneously. This does not factorise across loci (a match
requires agreement at every marker) or across candidates (they share
ancestors), so it is obtained by exact sampling from the same
inside-pass messages used above: each draw is an unweighted, exact
sample from the true posterior over every male's genotype, and averaging
the "does anyone match" indicator over many draws gives an unbiased
estimate with a reportable standard error - no importance weights, no
tuning parameter, no convergence criterion. See `--at-least-one` below.

### Mutation model

A symmetric stepwise mutation model with one- and two-step mutations is
used for each meiosis:

```
p(j -> k) = 1 - mu1 - mu2   if k == j
            mu1 / 2         if |k - j| == 1
            mu2 / 2         if |k - j| == 2
            0               otherwise
```

The user supplies the overall mutation rate `mu = mu1 + mu2` (default
`0.1`) and a two-step fraction `f` (`--two-step-frac`, default `0`), from
which `mu1 = (1 - f) * mu` and `mu2 = f * mu`; `f = 0` gives the classic
one-step model. Alleles are integers in a fixed range (default `[3, 30]`);
at the range boundaries the stay probability absorbs the mass that would
leave the range.

### Multi-copy markers

Loci such as DYS385 occur in `c > 1` copies; the lab reports only the
unordered allele combination (e.g. `10;11`). Internally each male's
genotype is an ordered copy vector, and an observation is consistent with
every ordering of it. Mutation acts independently on each copy.

`mu` is always the **per-copy rate** and is applied unchanged to every
copy (split into one- and two-step mutations with the fraction `f`); no
conversion takes place. The number of copies `c` is detected from the
input data. This convention is the same for `compute_match_probs_fast`,
`compute_match_probs_multicopy`, `compute_at_least_one_match` and the
CLI's `--mu`; for single-copy loci (`c = 1`) `mu` is simply the rate per
meiosis. For information, the CLI also prints the corresponding all-copy
rate of the locus, `mu_all = 1 - (1 - mu)^c`.

## Repository contents

| File | Description |
|---|---|
| `ystr_pedigree.py` | Core algorithm: pedigree data structures, CSV loading, mutation model, `compute_match_probs_fast` (linear-time inside-outside version, dispatching to `compute_match_probs_multicopy` for multi-copy loci), and `compute_at_least_one_match` (i-MP via exact sampling from the peeling messages). |
| `cli.py` | Command-line interface for running the calculator on a pedigree CSV file. |
| `runtime_analysis.py` | Empirical runtime benchmarking script that measures how `compute_match_probs_fast` scales with pedigree size, number of untyped males, allele range and number of loci (plus a two-copy overlay series per panel), and how the i-MP sampling estimator scales with the number of untyped candidates, and plots the results. |
| `compare_exact_vs_sampled.py` | Validates the shipped i-MP sampling estimator against an exact, non-sampling reference (full inclusion-exclusion) for the example pedigrees; only pedigrees with at least two candidates are reported. |
| `derive_multicopy_analytic.py` | Independently re-derives closed-form match-probability formulas for the two-copy pedigrees `A01_mc1`, `A01_mc2`, `B01_mc`, `C01_mc` and `C02_mc` and cross-checks them against `compute_match_probs_multicopy`. |
| `tests/` | `pytest` test suite. |
| `Pedigrees/` | Example pedigree CSV files: single-copy `A01`, `B01`, `C01`, `C02`, `D01`, `D02`, `E01`, `E02` and the realistic 15-male, 30-locus `G01`; two-copy (DYS385-like) `A01_mc1`, `A01_mc2`, `B01_mc`, `C01_mc`, `C02_mc` and `F01_mc`. |
| `LICENSE.txt` | MIT license. |

## Pedigree CSV format

Each pedigree is a CSV file with one row per male:

| Column | Description |
|---|---|
| `male_id` | Unique integer identifier. |
| `generation` | 0-based generation index (0 = suspect / most recent common ancestor). |
| `father_id` | `male_id` of the father, or `-1` for the root. |
| `allele_<LOCUS>` | Allele value at locus `LOCUS`. Use `0` for an untyped individual. For a multi-copy locus, give the alleles semicolon-separated (e.g. `10;11`). Add one such column per locus. |

Example (`Pedigrees/D01.csv`), a single locus `M1` with two untyped males
between the suspect and a typed descendant:

```csv
male_id,generation,father_id,allele_M1
1,0,-1,13
2,1,1,0
3,2,2,0
4,3,3,13
```

## Usage

### Command line

```bash
python cli.py Pedigrees/D01.csv --mu 0.1 --out results.csv
```

Arguments:
- `PEDIGREE_CSV` - path to the input pedigree CSV file.
- `--mu` - mutation rate per meiosis (default: `0.1`); for multi-copy
  loci this is the per-copy rate (see *Multi-copy markers*).
- `--two-step-frac` - fraction of mutations that are two-step (default: `0`).
- `--out` - optional path to write results as CSV instead of only printing them.
- `--at-least-one` - also compute i-MP, the probability that at least one
  untyped, non-excluded male anywhere in the pedigree matches the suspect
  at every locus simultaneously.
- `--n-samples` - number of posterior family draws used for `--at-least-one`
  (default: `200000`).
- `--seed` - random seed for `--at-least-one` (default: nondeterministic).

For each locus found in the file, the CLI prints the average match
probability across all untyped males, plus the per-male match
probability.

```bash
python cli.py Pedigrees/G01.csv --mu 0.1 --at-least-one --n-samples 200000 --seed 42
```

```
=== At least one match, i-MP (over 30 loci, 6.191 s) ===
    i-MP = 0.009430  (se = 2.16e-04, n_samples = 200000)
    candidates: 11
```

Only untyped males are i-MP candidates: a male typed at any of the loci
used is excluded, whether or not his genotype matches the suspect's,
because his match status is then observed rather than uncertain. If there
are no candidates, i-MP is reported as `0`. If every candidate's
individual match probability is far below `1/n_samples`, the sampling
estimate rounds to zero and the CLI instead reports the first-order
approximation `i-MP ≈ Σ_i P_i` (Section 3.2 of the technical note), with
no standard error.

For the special case where all candidates are conditionally independent
given the typed suspect (e.g. two sons with no shared untyped ancestor,
single locus), i-MP has a simple closed form you can check by hand:
`i-MP = 1 - Π_i (1 - P_i)`, with each `P_i` from `compute_match_probs_fast`
/ `compute_match_probs_multicopy`. In general, though, candidates can
share untyped ancestors, which is exactly why `compute_at_least_one_match`
samples rather than combines the per-individual probabilities directly.

### Runtime analysis

```bash
python runtime_analysis.py
```

Requires `numpy` and `matplotlib`. Benchmarks `compute_match_probs_fast`
across pedigree size, number of untyped males, allele range, and number
of loci, for three pedigree topologies (linear chain, and branching
factors of 3 and 5), each panel with an additional two-copy series
(branching factor 3). A fifth panel shows the runtime of the i-MP
sampling estimator (`n_samples = 200000`) against the number of untyped
candidates. Runtimes are wall-clock (`time.perf_counter`). The figure is
saved to `runtime_analysis.png` (plus `.pdf` and `.svg`).

### Validation

```bash
python derive_multicopy_analytic.py          # closed-form multi-copy checks
python compare_exact_vs_sampled.py --mu 0.1 --n-samples 1000000
python -m pytest tests
```

## Requirements

- Python 3.10+
- `numpy` (required by `ystr_pedigree.py` and `cli.py`, for multi-copy markers and i-MP sampling)
- `matplotlib` (only for `runtime_analysis.py`)
- `sympy` (only for `derive_multicopy_analytic.py`)
- `pytest` (only for the test suite)

## License

MIT — see [`LICENSE.txt`](LICENSE.txt).
