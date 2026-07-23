"""
ystr_pedigree.py
================
Exact pedigree-based Y-STR match probability calculator.

Implements the Elston-Stewart algorithm adapted for Y-chromosomal STRs.

Algorithm
---------
The pedigree is represented as a rooted tree of males, with the most
recent common ancestor (MRCA) / suspect at the root (index 0). Each
male carries either a known allele or is untyped (allele == 0).

Match probabilities are computed exactly via backward induction over
generations: for each male i, the conditional likelihood beta[i][k] is
the probability of the observed allele configuration in the subtree
rooted at i, given that i carries allele k. The final match probability
for an untyped male is the ratio P2/P1, where P1 is the likelihood of
the observed data and P2 is the likelihood when the untyped male is
hypothetically assigned the suspect's allele.

Mutation model
--------------
A one-step symmetric stepwise mutation model is used:

    p(j -> k) = 1 - mu   if k == j
                mu / 2   if |k - j| == 1
                0        otherwise

where mu is the locus-specific mutation rate per meiosis.
"""

from __future__ import annotations
 
import csv
from dataclasses import dataclass, field
from typing import Callable
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
MINALL: int   = 3      # Lower bound of allele range (inclusive)
MAXALL: int   = 30     # Upper bound of allele range (inclusive)
DEFAULT_MU: float = 0.1  # Default mutation rate per meiosis
 
 
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
 
@dataclass
class Male:
    """A single male in the pedigree.
 
    Parameters
    ----------
    male_id:
        Unique integer identifier.
    generation:
        Zero-based generation index (0 = MRCA / suspect).
    father_id:
        ``male_id`` of the father, or -1 for the root.
    alleles:
        Dict mapping locus name to integer allele value.
        Use 0 to indicate an untyped individual at a given locus.
    """
    male_id:    int
    generation: int
    father_id:  int
    alleles:    dict = field(default_factory=dict)
 
 
# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
 
def load_pedigree_from_csv(filepath: str) -> tuple[list[Male], list[str]]:
    """Load a pedigree from a CSV file.
 
    Expected columns
    ----------------
    male_id     : int  — unique individual ID
    generation  : int  — 0-based generation (0 = suspect / MRCA)
    father_id   : int  — father's male_id; use -1 (or any absent ID) for root
    allele_<LOC> : int  — allele at locus LOC; 0 = untyped
 
    Returns
    -------
    males : list[Male]
        Sorted by (generation, male_id).
    loci : list[str]
        Locus names inferred from column headers.
    """
    males: list[Male] = []
    loci:  list[str]  = []
 
    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or malformed CSV: {filepath}")
 
        loci = [
            col.replace("allele_", "")
            for col in reader.fieldnames
            if col.startswith("allele_")
        ]
 
        for row in reader:
            males.append(Male(
                male_id    = int(row["male_id"]),
                generation = int(row["generation"]),
                father_id  = int(row["father_id"]),
                alleles    = {
                    col.replace("allele_", ""): int(val)
                    for col, val in row.items()
                    if col.startswith("allele_")
                },
            ))
 
    males.sort(key=lambda m: (m.generation, m.male_id))
    return males, loci
 
 
# ---------------------------------------------------------------------------
# Mutation model
# ---------------------------------------------------------------------------
 
def make_stepwise_mutation_model(mu: float) -> Callable[[int, int], float]:
    """Return the one-step symmetric stepwise mutation probability function.
 
    Parameters
    ----------
    mu:
        Mutation rate per meiosis.
 
    Returns
    -------
    p(j, k) : float
        Probability that allele *j* mutates to allele *k* in one meiosis.
    """
    def p(j: int, k: int) -> float:
        diff = abs(k - j)
        if diff == 0:
            return 1.0 - mu
        elif diff == 1:
            return mu / 2.0
        else:
            return 0.0
    return p
 
 
# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_match_probs_fast(
    males:  list[Male],
    locus:  str,
    mu:     float = DEFAULT_MU,
    minall: int   = MINALL,
    maxall: int   = MAXALL,
) -> tuple[dict[int, float], float]:
    """Compute match probabilities for all untyped males at one locus.

    For each untyped male i, the match probability is

        MP_i = P(data | male i has suspect allele) / P(data)

    Uses a two-pass Elston-Stewart inside-outside sweep so that every
    untyped male's match probability is obtained in O(n*A) total, instead
    of re-peeling the whole tree once per untyped male.

    The "inside" pass computes ``beta[i][k]`` bottom-up: the likelihood of
    the subtree rooted at i, given i carries allele k. A second "outside"
    pass computes ``alpha[i][k]`` top-down: the combined likelihood
    contribution of everything *not* in i's subtree (ancestors and sibling
    subtrees), given i carries allele k. For any untyped male j and
    candidate allele s:

        P(all data AND j = s) = alpha[j][s] * beta[j][s]

    so every match probability is read off in O(1) once both O(n*A)
    passes are done.

    Parameters
    ----------
    males:
        Pedigree members sorted by (generation, male_id). A father must
        always appear before its sons (see ``load_pedigree_from_csv``).
    locus:
        Locus name.
    mu:
        Mutation rate per meiosis (default: DEFAULT_MU).
    minall, maxall:
        Inclusive allele range (default: MINALL, MAXALL).

    Returns
    -------
    match_probs : dict[int, float]
        Maps individual index -> match probability.
        Only untyped males are included.
    average : float
        Average match probability over all untyped males.
        Returns 0.0 if suspect is untyped or no untyped males exist.
    """
    base    = [m.alleles.get(locus, 0) for m in males]
    suspect = base[0]

    if suspect == 0:
        return {}, 0.0

    n = len(males)
    id_to_idx = {m.male_id: i for i, m in enumerate(males)}
    sons:   list[list[int]] = [[] for _ in range(n)]
    father: list[int]       = [-1] * n
    for i, m in enumerate(males):
        if m.father_id in id_to_idx:
            fi = id_to_idx[m.father_id]
            if fi >= i:
                raise ValueError(
                    "males must be sorted by (generation, male_id): "
                    f"father at index {fi} appears after child at index {i}"
                )
            sons[fi].append(i)
            father[i] = fi

    p            = make_stepwise_mutation_model(mu)
    allele_range = range(minall, maxall + 1)
    A            = maxall - minall + 1

    def idx(x: int) -> int:
        return x - minall

    # --- inside pass (beta), bottom-up ---------------------------------
    # message[i][*] is i's contribution to its father, indexed by the
    # father's candidate allele -- i.e. sum_ll p(k_father, ll) * beta[i][ll].
    beta    = [[0.0] * A for _ in range(n)]
    message = [[0.0] * A for _ in range(n)]

    for i in range(n - 1, -1, -1):
        e = base[i]
        if not sons[i]:
            if e == 0:
                for kk in range(A):
                    beta[i][kk] = 1.0
            else:
                beta[i][idx(e)] = 1.0
        else:
            for k in allele_range:
                if e != 0 and e != k:
                    continue
                val = 1.0
                for s in sons[i]:
                    val *= message[s][idx(k)]
                beta[i][idx(k)] = val

        for k_f in allele_range:
            lo  = max(minall, k_f - 1)
            hi  = min(maxall, k_f + 1)
            acc = 0.0
            for ll in range(lo, hi + 1):
                acc += p(k_f, ll) * beta[i][idx(ll)]
            message[i][idx(k_f)] = acc

    # --- outside pass (alpha), top-down ---------------------------------
    alpha = [[1.0] * A for _ in range(n)]  # root has nothing "outside" it

    for f in range(n):
        kids = sons[f]
        if not kids:
            continue

        m_sons = len(kids)
        msgs   = [message[s] for s in kids]

        # leave-one-out products over siblings, per candidate father allele
        prefix = [[1.0] * A for _ in range(m_sons + 1)]
        suffix = [[1.0] * A for _ in range(m_sons + 1)]
        for t in range(m_sons):
            row_prev, row_msg, row_next = prefix[t], msgs[t], prefix[t + 1]
            for kk in range(A):
                row_next[kk] = row_prev[kk] * row_msg[kk]
        for t in range(m_sons - 1, -1, -1):
            row_prev, row_msg, row_next = suffix[t + 1], msgs[t], suffix[t]
            for kk in range(A):
                row_next[kk] = row_prev[kk] * row_msg[kk]

        ef = base[f]
        father_factor = [0.0] * A
        for k_f in allele_range:
            kfi = idx(k_f)
            e_val = 1.0 if (ef == 0 or ef == k_f) else 0.0
            father_factor[kfi] = alpha[f][kfi] * e_val

        for t, i in enumerate(kids):
            outside_excl_i = [
                father_factor[kk] * prefix[t][kk] * suffix[t + 1][kk]
                for kk in range(A)
            ]
            for k_i in allele_range:
                lo  = max(minall, k_i - 1)
                hi  = min(maxall, k_i + 1)
                acc = 0.0
                for k_f in range(lo, hi + 1):
                    acc += p(k_f, k_i) * outside_excl_i[idx(k_f)]
                alpha[i][idx(k_i)] = acc

    P1 = beta[0][idx(suspect)]
    if P1 == 0.0:
        return {}, 0.0

    match_probs: dict[int, float] = {}
    for i, allele in enumerate(base):
        if allele == 0:
            P2 = alpha[i][idx(suspect)] * beta[i][idx(suspect)]
            match_probs[i] = P2 / P1

    average = sum(match_probs.values()) / len(match_probs) if match_probs else 0.0
    return match_probs, average
