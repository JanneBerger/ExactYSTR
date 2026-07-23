# Exact Y-STR Match Probabilities

Exact computation of Y-chromosomal STR (Y-STR) match probabilities within a
pedigree, using the Elston-Stewart algorithm.

Given a pedigree with a typed suspect and one or more relatives who are
untyped (missing) at a locus, this tool computes, for each untyped male, the
probability that he carries the suspect's allele - exactly, without Monte
Carlo simulation, accounting for mutation between generations.

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

### Mutation model

A one-step symmetric stepwise mutation model is used for each meiosis:

```
p(j -> k) = 1 - mu   if k == j
            mu / 2   if |k - j| == 1
            0        otherwise
```

where `mu` is the mutation rate per meiosis (default `0.1`), and alleles
are integers in a fixed range (default `[3, 30]`).

## Repository contents

| File | Description |
|---|---|
| `ystr_pedigree.py` | Core algorithm: pedigree data structures, CSV loading, mutation model, and `compute_match_probs_fast` (linear-time inside-outside version). |
| `cli.py` | Command-line interface for running the calculator on a pedigree CSV file. |
| `runtime_analysis.py` | Empirical runtime benchmarking script that measures how `compute_match_probs_fast` scales with pedigree size, number of untyped males, allele range, and number of loci, and plots the results. |
| `Pedigrees/` | Example pedigree CSV files (`A01.csv` … `E02.csv`). |
| `LICENSE.txt` | MIT license. |

## Pedigree CSV format

Each pedigree is a CSV file with one row per male:

| Column | Description |
|---|---|
| `male_id` | Unique integer identifier. |
| `generation` | 0-based generation index (0 = suspect / most recent common ancestor). |
| `father_id` | `male_id` of the father, or `-1` for the root. |
| `allel_<LOCUS>` | Allele value at locus `LOCUS`. Use `0` for an untyped individual. Add one such column per locus. |

Example (`Pedigrees/D01.csv`), a single locus `M1` with two untyped males
between the suspect and a typed descendant:

```csv
male_id,generation,father_id,allel_M1
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
- `--mu` - mutation rate per meiosis (default: `0.1`).
- `--out` - optional path to write results as CSV instead of only printing them.

For each locus found in the file, the CLI prints the average match
probability across all untyped males, plus the per-male match
probability.

### As a library

```python
from ystr_pedigree import load_pedigree_from_csv, compute_match_probs_fast

males, loci = load_pedigree_from_csv("Pedigrees/D01.csv")
match_probs, average = compute_match_probs_fast(males, "M1", mu=0.1)

for i, prob in match_probs.items():
    print(males[i].male_id, prob)
```

`compute_match_probs_fast` computes all match probabilities for a locus in
a single linear-time inside-outside sweep, instead of re-peeling the whole
tree once per untyped male.

### Runtime analysis

```bash
python runtime_analysis.py
```

Requires `numpy` and `matplotlib`. Benchmarks `compute_match_probs_fast`
across pedigree size, number of untyped males, allele range, and number
of loci, for three pedigree topologies (linear chain, and branching
factors of 3 and 5), and saves a 4-panel figure to
`runtime_analysis.png`.

## Requirements

- Python 3.10+ (standard library only, for `ystr_pedigree.py` and `cli.py`)
- `numpy` and `matplotlib` (only for `runtime_analysis.py`)

## License

MIT — see [`LICENSE.txt`](LICENSE.txt).
