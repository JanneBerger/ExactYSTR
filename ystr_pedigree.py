"""
ystr_pedigree.py
================
Exact pedigree-based Y-STR match probability calculator.

Implements the Elston-Stewart algorithm adapted for Y-chromosomal STRs,
including exact handling of multi-copy markers (e.g. DYS385).

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
A stepwise mutation model is used (one-step by default, two-step optional):

    p(j -> k) = 1 - mu - mu2   if k == j
                mu / 2          if |k - j| == 1
                mu2 / 2         if |k - j| == 2
                0               otherwise

where mu is the one-step rate and mu2 the two-step rate per meiosis.
Setting mu2 = 0 (default) recovers the standard one-step SSM.

Multi-copy markers
------------------
Some Y-STR loci appear more than once on the chromosome (e.g. DYS385
has c=2 copies). The lab reports an unordered set {x1, x2, ...}.

An ordered latent state x = (x1, ..., xc) is maintained internally;
the order is discarded only at the emission step:

    E_g(x) = 1  if sort(x) == g,  else 0

The transition kernel on ordered states is the c-fold tensor product

    T^(c)(x -> y) = prod_a T(x_a -> y_a)

which inherits the symmetry and reversibility of T.  Messages are
computed by sequential contraction over copy axes at cost O(c * A^c)
rather than the naive O(A^{2c}).

The all-copy mutation rate mu_all reported by a kit is converted to the
per-copy rate

    mu = 1 - (1 - mu_all)^(1/c)

before being used in the kernel.

CSV format for multi-copy alleles
----------------------------------
Single-copy locus:  allele_DYS393 = 13
Multi-copy locus:   allele_DYS385 = 10;11   (semicolon-separated, sorted)
Untyped:            allele_<LOC>  = 0        (single integer 0 always)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINALL: int   = 3      # Lower bound of allele range (inclusive)
MAXALL: int   = 30     # Upper bound of allele range (inclusive)
DEFAULT_MU: float  = 0.1   # Default one-step mutation rate per meiosis
DEFAULT_MU2: float = 0.0   # Default two-step mutation rate per meiosis
_WINDOW_MARGIN: int = 5  # Allele window half-width for multi-copy markers


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
        Dict mapping locus name to allele value.
        Single-copy locus: int (0 = untyped).
        Multi-copy locus:  tuple[int, ...] sorted, e.g. (10, 11).
        Untyped (any):     0 or (0,).
    """
    male_id:    int
    generation: int
    father_id:  int
    alleles:    dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _parse_allele_cell(val: str) -> int | tuple[int, ...]:
    """Parse one allele CSV cell.

    ``"13"``      -> ``13``   (single-copy)
    ``"10;11"``   -> ``(10, 11)``  (multi-copy, two values)
    ``"0"``       -> ``0``   (untyped, any copy count)
    """
    val = val.strip()
    if ";" in val:
        return tuple(int(v.strip()) for v in val.split(";"))
    return int(val)


def load_pedigree_from_csv(filepath: str) -> tuple[list[Male], list[str]]:
    """Load a pedigree from a CSV file.

    Expected columns
    ----------------
    male_id     : int  — unique individual ID
    generation  : int  — 0-based generation (0 = suspect / MRCA)
    father_id   : int  — father's male_id; use -1 (or any absent ID) for root
    allele_<LOC> : int or semicolon-separated ints — allele at locus LOC
                   Single-copy:  ``13``
                   Multi-copy:   ``10;11``
                   Untyped:      ``0``

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
                    col.replace("allele_", ""): _parse_allele_cell(val)
                    for col, val in row.items()
                    if col.startswith("allele_")
                },
            ))

    males.sort(key=lambda m: (m.generation, m.male_id))
    return males, loci


# ---------------------------------------------------------------------------
# Mutation model
# ---------------------------------------------------------------------------

def make_stepwise_mutation_model(mu: float, mu2: float = 0.0) -> Callable[[int, int], float]:
    """Return the stepwise mutation probability function (one- and two-step).

    Parameters
    ----------
    mu:
        One-step mutation rate per meiosis.
    mu2:
        Two-step mutation rate per meiosis (default 0 → one-step SSM only).

    Returns
    -------
    p(j, k) : float
        Probability that allele *j* mutates to allele *k* in one meiosis.
    """
    def p(j: int, k: int) -> float:
        diff = abs(k - j)
        if diff == 0:
            return 1.0 - mu - mu2
        elif diff == 1:
            return mu / 2.0
        elif diff == 2:
            return mu2 / 2.0
        else:
            return 0.0
    return p


# ---------------------------------------------------------------------------
# Multi-copy helpers
# ---------------------------------------------------------------------------

def _infer_copies(males: list[Male], locus: str) -> int:
    """Return number of copies for *locus* (1 for single-copy)."""
    for m in males:
        a = m.alleles.get(locus, 0)
        if isinstance(a, tuple) and any(v != 0 for v in a):
            return len(a)
    return 1


def _to_genotype(a: int | tuple[int, ...]) -> tuple[int, ...] | None:
    """Normalise a raw allele value to a sorted tuple, or None if untyped."""
    if isinstance(a, tuple):
        if all(v == 0 for v in a):
            return None
        return tuple(sorted(a))
    return None if a == 0 else (a,)


def _emission_mask(g: tuple[int, ...] | None, shape: tuple[int, ...],
                   win_min: int) -> np.ndarray:
    """Boolean emission array for multi-copy ordered state space.

    Returns an array of shape ``shape = (A,)*c`` with 1.0 where
    ``sort(ordered_state) == g``, and 1.0 everywhere if ``g is None``
    (untyped individual).
    """
    if g is None:
        return np.ones(shape, dtype=np.float64)
    A = shape[0]
    win_max = win_min + A - 1
    mask = np.zeros(shape, dtype=np.float64)
    if not all(win_min <= v <= win_max for v in g):
        return mask  # genotype outside window -> zero emission
    g_idx = tuple(v - win_min for v in tuple(sorted(g)))
    for perm in set(permutations(g_idx)):
        mask[perm] = 1.0
    return mask


def _build_T(mu: float, A: int, mu2: float = 0.0) -> np.ndarray:
    """Build the A×A stepwise mutation transition matrix (one- and two-step).

    ``T[j, k]`` is the probability that an allele at offset *j* from
    ``win_min`` mutates to offset *k* in one meiosis.  The diagonal is set
    so each row sums to 1 (automatically handles boundary alleles).
    """
    T = np.zeros((A, A))
    for j in range(A):
        if j > 0:
            T[j, j - 1] = mu / 2.0
        if j < A - 1:
            T[j, j + 1] = mu / 2.0
        if j > 1:
            T[j, j - 2] = mu2 / 2.0
        if j < A - 2:
            T[j, j + 2] = mu2 / 2.0
        T[j, j] = 1.0 - T[j, :].sum()
    return T


def _compute_message(beta: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute the message from a child node to its father.

    ``msg[xf] = Σ_y  T^(c)(xf -> y)  beta[y]``

    Uses sequential contraction over the *c* copy axes at cost
    ``O(c * A^c)`` rather than the naive ``O(A^(2c))``.

    Because the SSM kernel is symmetric (``T[i,j] == T[j,i]``), this
    same function is also used for the downward (alpha) messages.

    Parameters
    ----------
    beta : np.ndarray, shape ``(A,)*c``
        Child's beta (or outside-excl-i) array.
    T : np.ndarray, shape ``(A, A)``
        Per-copy transition matrix.

    Returns
    -------
    np.ndarray, shape ``(A,)*c``
        Message array indexed by the *parent's* ordered state.
    """
    result = beta
    for _ in range(beta.ndim):
        # Contract T's child-axis (axis 1) against the current leading
        # child axis of `result` (axis 0).
        result = np.tensordot(T, result, axes=([1], [0]))
        # Move the freshly created parent axis from front to back so
        # the remaining child axes stay at the front for the next step.
        result = np.moveaxis(result, 0, -1)
    return result


# ---------------------------------------------------------------------------
# Core algorithm — multi-copy loci
# ---------------------------------------------------------------------------

def compute_match_probs_multicopy(
    males:         list[Male],
    locus:         str,
    n_copies:      int,
    mu_all:        float = DEFAULT_MU,
    mu2_all:       float = DEFAULT_MU2,
    minall:        int   = MINALL,
    maxall:        int   = MAXALL,
    window_margin: int   = _WINDOW_MARGIN,
) -> tuple[dict[int, float], float]:
    """Exact match probabilities for a multi-copy locus.

    Implements the ordered-state Elston-Stewart sweep described in:
    "Exact pedigree-based Y-STR match probabilities: how the multi-copy
    and 'at least one match' obstacles were removed" (2026).

    The internal state is an ordered copy-vector ``x = (x_1, ..., x_c)``.
    Unordering is applied only at the emission step, which restores the
    reversibility of the mutation process and allows the standard
    inside-outside peeling to proceed.

    Parameters
    ----------
    males : list[Male]
        Pedigree sorted by (generation, male_id).  Father must appear
        before all sons (as guaranteed by ``load_pedigree_from_csv``).
    locus : str
        Locus name.  Allele values must be tuples of ints (multi-copy)
        or 0 for untyped.
    n_copies : int
        Number of chromosomal copies *c*.
    mu_all : float
        All-copy mutation rate per meiosis as reported by the kit.
        Converted internally to per-copy rate
        ``mu = 1 - (1 - mu_all)**(1/c)``.
    minall, maxall : int
        Global allele bounds; may be tightened by *window_margin*.
    window_margin : int
        Half-width of the allele window centred on observed alleles.
        Reducing the window shrinks the state space from
        ``(maxall - minall + 1)^c`` to roughly ``(2*window_margin)^c``,
        making *c = 3* and *c = 4* copy-counts practical.

    Returns
    -------
    match_probs : dict[int, float]
        Maps individual index -> match probability.  Only untyped males
        are included.
    average : float
        Average match probability over all untyped males.
        Returns 0.0 if the suspect is untyped or there are no untyped
        males.
    """
    c = n_copies

    # Per-copy mutation rates (equation 3 of the technical note)
    mu  = 1.0 - (1.0 - mu_all)  ** (1.0 / c)
    mu2 = 1.0 - (1.0 - mu2_all) ** (1.0 / c)

    # Normalise allele values
    base: list[tuple[int, ...] | None] = [
        _to_genotype(m.alleles.get(locus, 0)) for m in males
    ]
    suspect_g = base[0]
    if suspect_g is None:
        return {}, 0.0

    # Narrow allele window to observed values ± margin
    obs_vals = [v for g in base if g is not None for v in g]
    if not obs_vals:
        return {}, 0.0
    win_min = max(minall, min(obs_vals) - window_margin)
    win_max = min(maxall, max(obs_vals) + window_margin)
    A = win_max - win_min + 1

    T = _build_T(mu, A, mu2)

    # Pedigree connectivity
    n        = len(males)
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

    shape    = (A,) * c
    em_cache: dict[tuple[int, ...] | None, np.ndarray] = {}

    def em(g: tuple[int, ...] | None) -> np.ndarray:
        if g not in em_cache:
            em_cache[g] = _emission_mask(g, shape, win_min)
        return em_cache[g]

    # --- inside pass (beta), bottom-up ---
    beta:    list[np.ndarray] = [np.zeros(shape) for _ in range(n)]
    message: list[np.ndarray] = [np.zeros(shape) for _ in range(n)]

    for i in range(n - 1, -1, -1):
        g = base[i]
        if not sons[i]:
            beta[i] = em(g).copy()
        else:
            val = em(g).copy()
            for s in sons[i]:
                val *= message[s]
            beta[i] = val

        message[i] = _compute_message(beta[i], T)

    # --- outside pass (alpha), top-down ---
    alpha: list[np.ndarray] = [np.ones(shape) for _ in range(n)]

    for f in range(n):
        kids = sons[f]
        if not kids:
            continue

        m_sons = len(kids)
        msgs   = [message[s] for s in kids]

        # Leave-one-out prefix/suffix products (per parent ordered state)
        prefix = [np.ones(shape) for _ in range(m_sons + 1)]
        suffix = [np.ones(shape) for _ in range(m_sons + 1)]
        for t in range(m_sons):
            prefix[t + 1] = prefix[t] * msgs[t]
        for t in range(m_sons - 1, -1, -1):
            suffix[t] = suffix[t + 1] * msgs[t]

        # Father's combined outside contribution (alpha × emission)
        father_factor = alpha[f] * em(base[f])

        for t, i in enumerate(kids):
            outside_excl_i = father_factor * prefix[t] * suffix[t + 1]
            # By SSM symmetry T[j,k] == T[k,j], the downward message
            # alpha[i][yi] = Σ_{xf} T^c(xf→yi) * outside_excl_i[xf]
            # equals _compute_message(outside_excl_i)[yi].
            alpha[i] = _compute_message(outside_excl_i, T)

    # --- match probabilities ---
    suspect_mask = em(suspect_g)
    P1 = float(np.sum(beta[0] * suspect_mask))
    if P1 == 0.0:
        return {}, 0.0

    match_probs: dict[int, float] = {}
    for i, g in enumerate(base):
        if g is None:  # untyped
            P2 = float(np.sum(alpha[i] * beta[i] * suspect_mask))
            match_probs[i] = P2 / P1

    average = sum(match_probs.values()) / len(match_probs) if match_probs else 0.0
    return match_probs, average


# ---------------------------------------------------------------------------
# Core algorithm — single-copy loci (original fast implementation)
# ---------------------------------------------------------------------------

def compute_match_probs_fast(
    males:  list[Male],
    locus:  str,
    mu:     float = DEFAULT_MU,
    mu2:    float = DEFAULT_MU2,
    minall: int   = MINALL,
    maxall: int   = MAXALL,
) -> tuple[dict[int, float], float]:
    """Compute match probabilities for all untyped males at one locus.

    Automatically dispatches to :func:`compute_match_probs_multicopy`
    when the locus has more than one copy (detected from tuple-valued
    alleles in *males*).

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
        One-step mutation rate per meiosis (default: DEFAULT_MU).
        For multi-copy loci this is interpreted as the all-copy rate.
    mu2:
        Two-step mutation rate per meiosis (default: 0).
        For multi-copy loci also interpreted as the all-copy rate.
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
    # Dispatch to multi-copy implementation when needed
    n_copies = _infer_copies(males, locus)
    if n_copies > 1:
        return compute_match_probs_multicopy(
            males, locus, n_copies, mu_all=mu, mu2_all=mu2, minall=minall, maxall=maxall
        )

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

    p            = make_stepwise_mutation_model(mu, mu2)
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
            lo  = max(minall, k_f - 2)
            hi  = min(maxall, k_f + 2)
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
                lo  = max(minall, k_i - 2)
                hi  = min(maxall, k_i + 2)
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
