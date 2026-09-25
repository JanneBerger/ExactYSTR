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

The overall (all-copy) mutation rate mu_all of the locus, as usually
reported for multi-copy markers, is converted to the per-copy rate

    mu = 1 - (1 - mu_all)^(1/c)

before being used in the kernel.  With two-step mutations, the conversion
is applied to the total rate mu_all = mu1_all + mu2_all, and the per-copy
rate is split in the same one-/two-step proportion.  Every public entry
point (compute_match_probs_fast, compute_match_probs_multicopy,
compute_at_least_one_match and the CLI's --mu) takes this overall
per-locus rate; for single-copy loci (c=1) it is simply the per-meiosis
rate.

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

@dataclass
class _MulticopyInside:
    """Shared pedigree connectivity + inside (beta) pass for one locus.

    ``c == 1`` covers ordinary single-copy loci as a special case (the
    "ordered copy-vector" is then just a 1-tuple), so this same structure
    also backs the exact-sampling code in :func:`sample_locus_matches`
    for single-copy markers.
    """
    c:         int
    A:         int
    win_min:   int
    shape:     tuple[int, ...]
    T:         np.ndarray
    base:      list[tuple[int, ...] | None]
    suspect_g: tuple[int, ...]
    n:         int
    sons:      list[list[int]]
    father:    list[int]
    beta:      list[np.ndarray]
    message:   list[np.ndarray]
    em:        Callable[[tuple[int, ...] | None], np.ndarray]


def _multicopy_inside_pass(
    males:         list[Male],
    locus:         str,
    n_copies:      int,
    mu_all:        float = DEFAULT_MU,
    mu2_all:       float = DEFAULT_MU2,
    minall:        int   = MINALL,
    maxall:        int   = MAXALL,
) -> _MulticopyInside | None:
    """Build the pedigree tree and run the inside (beta) sweep for *locus*.

    This is the setup shared by :func:`compute_match_probs_multicopy`
    (which adds the outside/alpha pass for per-individual match
    probabilities) and :func:`sample_locus_matches` (which adds the
    downward/backward sampling pass for the "at least one match"
    probability). See the module docstring and Sections 2 and 3.2 of
    the technical note.

    Returns ``None`` if the suspect is untyped at *locus*, or if no
    allele has been observed at all (mirrors the early-exit conditions
    of :func:`compute_match_probs_multicopy`).
    """
    c = n_copies

    # Per-copy mutation rates (equation 3 of the technical note): the
    # overall per-locus rate mu_all + mu2_all is converted to a per-copy
    # rate, which is then split in the same one-/two-step proportion, so
    # the two-step fraction is preserved exactly on every copy.
    mu_tot_all = mu_all + mu2_all
    mu_tot = 1.0 - (1.0 - mu_tot_all) ** (1.0 / c)
    if mu_tot_all > 0.0:
        mu  = mu_tot * mu_all  / mu_tot_all
        mu2 = mu_tot * mu2_all / mu_tot_all
    else:
        mu = mu2 = 0.0

    # Normalise allele values
    base: list[tuple[int, ...] | None] = [
        _to_genotype(m.alleles.get(locus, 0)) for m in males
    ]
    suspect_g = base[0]
    if suspect_g is None:
        return None

    # Use the full allele range so that the calculation is exact rather
    # than an approximation restricted to a window around the observed
    # alleles.
    obs_vals = [v for g in base if g is not None for v in g]
    if not obs_vals:
        return None
    win_min = minall
    win_max = maxall
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

    return _MulticopyInside(
        c=c, A=A, win_min=win_min, shape=shape, T=T, base=base,
        suspect_g=suspect_g, n=n, sons=sons, father=father,
        beta=beta, message=message, em=em,
    )


def compute_match_probs_multicopy(
    males:         list[Male],
    locus:         str,
    n_copies:      int,
    mu_all:        float = DEFAULT_MU,
    mu2_all:       float = DEFAULT_MU2,
    minall:        int   = MINALL,
    maxall:        int   = MAXALL,
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
        All-copy one-step mutation rate per meiosis, as reported for
        the locus as a whole.
    mu2_all : float
        All-copy two-step mutation rate per meiosis (default 0).  The
        total ``mu_all + mu2_all`` is converted internally to the
        per-copy rate ``1 - (1 - (mu_all + mu2_all))**(1/c)``, which is
        split in the same one-/two-step proportion.
    minall, maxall : int
        Global allele bounds. The full range is used (no window
        narrowing around observed alleles), so the state space is
        exactly ``(maxall - minall + 1)^c`` and the result is exact,
        not an approximation.

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
    ctx = _multicopy_inside_pass(
        males, locus, n_copies, mu_all, mu2_all, minall, maxall
    )
    if ctx is None:
        return {}, 0.0

    n, sons        = ctx.n, ctx.sons
    beta, message  = ctx.beta, ctx.message
    em, T          = ctx.em, ctx.T
    base, shape    = ctx.base, ctx.shape

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
    suspect_mask = em(ctx.suspect_g)
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
    minall: int   = MINALL,
    maxall: int   = MAXALL,
    *,
    mu2:    float = DEFAULT_MU2,
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


# ---------------------------------------------------------------------------
# Core algorithm — "at least one match" (i-MP), Section 3 of the technical
# note: exact sampling from the peeling messages
# ---------------------------------------------------------------------------

def _sample_conditional(
    parent_idx: np.ndarray,
    beta:       np.ndarray,
    T:          np.ndarray,
    c:          int,
    A:          int,
    rng:        np.random.Generator,
) -> np.ndarray:
    """Draw one child state per sample from equation (18).

    For every sample, decode the father's *flat* ordered-state index,
    form the child's unnormalised distribution

        T^(c)(x_father -> y) * beta_child(y)

    as the outer product of the per-copy mutation rows ``T[x_a, :]``
    (the kernel factorises over copy-axes for a *fixed* parent state,
    Eq. 7) times *beta*, and draw ``y`` from it. Samples that share the
    same father state are batched into a single vectorised draw.

    Parameters
    ----------
    parent_idx : np.ndarray, shape (n_samples,)
        Flat index (into a length-``A**c`` array) of each sample's
        already-drawn father state.
    beta : np.ndarray, shape ``(A,)*c``
        The child's inside-pass beta array.
    T : np.ndarray, shape (A, A)
        Per-copy mutation transition matrix.
    c, A : int
        Copy count and per-copy allele-window width.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Flat index of the sampled child state, one per sample.
    """
    n_samples = parent_idx.shape[0]
    W = A ** c
    out = np.empty(n_samples, dtype=np.int64)

    unique_vals, inverse = np.unique(parent_idx, return_inverse=True)
    for u_pos, xf_flat in enumerate(unique_vals):
        xf = np.unravel_index(int(xf_flat), (A,) * c)
        vec = T[xf[0], :]
        for a in range(1, c):
            vec = np.multiply.outer(vec, T[xf[a], :])
        unnorm = (vec * beta).reshape(W)
        total = unnorm.sum()
        if total <= 0.0:
            # Numerical safety net only: with P1 > 0 every state reachable
            # under T from an already-sampled ancestor has positive beta,
            # so this branch should not trigger in practice.
            unnorm = beta.reshape(W).copy()
            total = unnorm.sum()
        probs = unnorm / total
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0  # guard against floating-point round-off

        mask = inverse == u_pos
        u = rng.random(int(mask.sum()))
        out[mask] = np.clip(np.searchsorted(cdf, u, side="right"), 0, W - 1)

    return out


def sample_locus_matches(
    males:         list[Male],
    locus:         str,
    mu_all:        float = DEFAULT_MU,
    mu2_all:       float = DEFAULT_MU2,
    n_copies:      int | None = None,
    minall:        int   = MINALL,
    maxall:        int   = MAXALL,
    n_samples:     int   = 1,
    rng:           np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Draw exact posterior samples of every male's genotype at *locus*.

    Implements the backward-sampling recursion of Section 3.2 of the
    technical note ("Route that works: exact sampling from the peeling
    messages"): the inside (beta) pass is run once, and then *n_samples*
    independent downward draws are taken,

        P(A_root = x)             ∝ beta_root(x)                (eq. 17)
        P(A_child = y | A_father) ∝ T(A_father -> y) beta_child(y)  (eq. 18)

    Each draw is an exact, unweighted sample from the true joint
    posterior of every male's genotype at *locus* given everything
    observed at *locus* -- no importance weights, no tuning parameter,
    no burn-in. This is what makes the union event needed for i-MP
    (equation 12) tractable: markers and individuals no longer need to
    be handled analytically together, they are simply read off jointly
    from one consistent set of sampled family trees.

    Because the internal state is the same ordered copy-vector used by
    :func:`compute_match_probs_multicopy`, this works unchanged for
    single-copy loci (``n_copies == 1``) and multi-copy loci alike.

    Returns
    -------
    matches : np.ndarray[bool], shape (n_samples, len(males))
        ``matches[s, i]`` is True iff male *i* is untyped at *locus* and
        his sampled genotype in draw *s* equals the suspect's genotype.
        For males *typed* at *locus* the column is left at its default
        of True (their match status there is deterministic -- see
        ``excluded`` -- so this is a neutral factor when AND-reducing
        match indicators across several loci).
    excluded : np.ndarray[bool], shape (len(males),)
        True for males typed at *locus* whose observed genotype differs
        from the suspect's. Deterministic (independent of sampling):
        such a male can never contribute a match at this locus.
    usable : bool
        False if the suspect is untyped at *locus*, or the observed
        data has zero likelihood (P1 == 0) -- the locus carries no
        usable information and should be skipped by the caller.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_copies is None:
        n_copies = _infer_copies(males, locus)

    n = len(males)
    ctx = _multicopy_inside_pass(
        males, locus, n_copies, mu_all, mu2_all, minall, maxall
    )
    if ctx is None:
        return np.zeros((n_samples, n), dtype=bool), np.zeros(n, dtype=bool), False

    P1 = float(np.sum(ctx.beta[0] * ctx.em(ctx.suspect_g)))
    if P1 == 0.0:
        return np.zeros((n_samples, n), dtype=bool), np.zeros(n, dtype=bool), False

    suspect_sorted = np.array(sorted(ctx.suspect_g)) - ctx.win_min
    excluded = np.zeros(n, dtype=bool)
    for i, g in enumerate(ctx.base):
        if g is not None and tuple(sorted(g)) != tuple(sorted(ctx.suspect_g)):
            excluded[i] = True

    W = ctx.A ** ctx.c
    matches = np.ones((n_samples, n), dtype=bool)
    sampled_idx: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * n

    for i in range(n):
        f = ctx.father[i]
        if f == -1:
            # Root: P(A_root = x) ∝ beta_root(x)  (eq. 17). The suspect's
            # own emission is already folded into beta_root, so this is
            # automatically restricted to states consistent with him.
            beta_flat = ctx.beta[i].reshape(W)
            probs = beta_flat / beta_flat.sum()
            cdf = np.cumsum(probs)
            cdf[-1] = 1.0
            u = rng.random(n_samples)
            idx = np.clip(np.searchsorted(cdf, u, side="right"), 0, W - 1)
        else:
            idx = _sample_conditional(
                sampled_idx[f], ctx.beta[i], ctx.T, ctx.c, ctx.A, rng
            )
        sampled_idx[i] = idx

        if ctx.base[i] is None:  # untyped here -> record this draw's match status
            coords = np.stack(np.unravel_index(idx, ctx.shape), axis=-1)  # (n_samples, c)
            sorted_coords = np.sort(coords, axis=-1)
            matches[:, i] = np.all(sorted_coords == suspect_sorted[np.newaxis, :], axis=-1)

    return matches, excluded, True


@dataclass
class AtLeastOneMatchResult:
    """Result of :func:`compute_at_least_one_match`.

    Attributes
    ----------
    i_mp : float
        The estimated "at least one match" probability (equation 12):
        the probability that *some* non-excluded, untyped male in the
        pedigree matches the suspect at every locus in ``loci_used``
        simultaneously.
    se : float | None
        Monte-Carlo standard error of ``i_mp``, ``sqrt(p(1-p)/n_samples)``
        (equation 19). ``None`` when ``i_mp`` was obtained analytically
        instead (``used_fallback`` True, or no candidates), since those
        values carry no sampling error.
    n_samples : int
        Number of independent posterior family draws used.
    loci_used : list[str]
        Loci that actually contributed (the suspect must be typed, and
        the observed data must have nonzero likelihood, at each).
    candidates : list[int]
        ``male_id`` of every candidate male: untyped at *every* locus in
        ``loci_used`` (a male typed at even one such locus is already
        either ruled out, if mismatching, or a known match there, and
        either way contributes no uncertainty) -- the set ``C`` of
        equation (12). i-MP is therefore, by construction, a statement
        about untyped relatives only, never about matches already
        established by typing.
    candidate_probs : dict[int, float]
        ``male_id -> P_i``, the exact individual match probability
        (product over ``loci_used`` of the per-locus match probability),
        i.e. the terms of the first-order inclusion-exclusion sum
        (equation 13).
    used_fallback : bool
        True if the sampling estimate rounded to exactly zero (it cannot
        resolve probabilities below ``1/n_samples``) and ``i_mp`` was
        instead taken from the first-order approximation
        ``i_mp ≈ sum(candidate_probs.values())``, which is accurate
        precisely in that regime (Section 3.2, "One limitation, with a
        clean remedy").
    fallback_estimate : float | None
        The first-order estimate, always computed for reference even
        when not used as ``i_mp``.
    """
    i_mp:              float
    se:                float | None
    n_samples:         int
    loci_used:         list[str]
    candidates:        list[int]
    candidate_probs:   dict[int, float]
    used_fallback:     bool
    fallback_estimate: float | None


def compute_at_least_one_match(
    males:         list[Male],
    loci:          list[str],
    mu:            float = DEFAULT_MU,
    two_step_frac: float = 0.0,
    n_samples:     int   = 200_000,
    minall:        int   = MINALL,
    maxall:        int   = MAXALL,
    seed:          int | None = None,
) -> AtLeastOneMatchResult:
    """Exact-sampling estimate of i-MP, the "at least one match" probability.

    Implements Section 3 of the technical note, "Obstacle 2: 'at least
    one match'". Let ``t`` be the suspect's profile and ``C`` the set of
    untyped, non-excluded candidates. i-MP is

        i-MP = P(exists i in C: male i matches t at every locus)      (12)

    This does not factorise across loci (a match requires agreement at
    *every* marker) or across candidates (they share ancestors), so it
    cannot be obtained from the per-locus, per-individual match
    probabilities alone. Instead, for each locus, one exact posterior
    sample of the *whole* pedigree's genotypes is drawn (via
    :func:`sample_locus_matches`) for each of ``n_samples`` independent
    draws; because markers mutate independently (equation 1), combining
    one independent draw per locus for each draw index yields an exact
    joint draw of every candidate's per-locus match status. AND-ing
    across loci gives each candidate's full-profile match indicator for
    that draw, OR-ing across candidates gives the "at least one match"
    indicator, and averaging over draws gives an unbiased estimate of
    i-MP with no importance weights and no convergence criterion
    (equation 19).

    When every candidate's individual match probability is minute, the
    sampling estimate can round to zero (it cannot resolve probabilities
    below ``1/n_samples``); in that regime the first-order
    inclusion-exclusion term ``i-MP ≈ sum_i P_i`` (equation 13) is used
    instead, exactly as prescribed in Section 3.2.

    Parameters
    ----------
    males, minall, maxall :
        As elsewhere in this module.
    loci : list[str]
        Loci to combine; a locus is skipped (and omitted from
        ``loci_used``) if the suspect is untyped there.
    mu : float
        Overall per-locus mutation rate (matching the CLI's ``--mu``).
        For multi-copy loci this is the all-copy rate ``mu_all``; the
        per-copy rate ``1 - (1 - mu_all)**(1/c)`` is derived
        automatically for each locus from its copy count.
    two_step_frac : float
        Fraction of each locus's mutations that are two-step
        (``mu1 = (1 - f) * mu``, ``mu2 = f * mu``).
    n_samples : int
        Number of independent posterior family draws.
    seed : int | None
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    AtLeastOneMatchResult
    """
    rng = np.random.default_rng(seed)
    n = len(males)

    excluded      = np.zeros(n, dtype=bool)
    typed_anywhere = np.zeros(n, dtype=bool)
    joint_matches = np.ones((n_samples, n), dtype=bool)
    per_locus_probs: dict[str, dict[int, float]] = {}
    loci_used: list[str] = []

    for locus in loci:
        c = _infer_copies(males, locus)
        lmu1 = mu * (1.0 - two_step_frac)
        lmu2 = mu * two_step_frac

        matches, excl, usable = sample_locus_matches(
            males, locus, mu_all=lmu1, mu2_all=lmu2, n_copies=c,
            minall=minall, maxall=maxall,
            n_samples=n_samples, rng=rng,
        )
        if not usable:
            continue

        loci_used.append(locus)
        excluded |= excl
        joint_matches &= matches
        typed_anywhere |= np.array([
            _to_genotype(m.alleles.get(locus, 0)) is not None for m in males
        ])

        mp, _avg = compute_match_probs_fast(
            males, locus, lmu1, mu2=lmu2, minall=minall, maxall=maxall
        )
        per_locus_probs[locus] = mp

    if not loci_used:
        return AtLeastOneMatchResult(
            i_mp=0.0, se=None, n_samples=n_samples, loci_used=[],
            candidates=[], candidate_probs={}, used_fallback=False,
            fallback_estimate=None,
        )

    joint_matches &= ~excluded[np.newaxis, :]

    # A candidate must be untyped at *every* locus in loci_used: a male
    # typed at even one such locus contributes no uncertainty there (he
    # is either ruled out, if mismatching, or already a known match),
    # so i-MP is a statement about untyped relatives only.
    candidate_mask = ~typed_anywhere
    candidate_mask[0] = False  # the suspect is never his own candidate
    candidates = [i for i in range(n) if candidate_mask[i]]

    candidate_probs: dict[int, float] = {
        males[i].male_id: float(np.prod([
            per_locus_probs[locus][i] for locus in loci_used
        ]))
        for i in candidates
    }

    if not candidates:
        return AtLeastOneMatchResult(
            i_mp=0.0, se=0.0,
            n_samples=n_samples, loci_used=loci_used,
            candidates=[], candidate_probs={},
            used_fallback=False, fallback_estimate=None,
        )

    any_match = joint_matches[:, candidates].any(axis=1)
    p_hat = float(any_match.mean())

    used_fallback = False
    fallback_estimate = sum(candidate_probs.values())
    if p_hat == 0.0:
        # Sampling resolution limit (Section 3.2): with n samples, i-MP
        # below ~1/n reads as exactly zero. The individual probabilities
        # are then tiny and effectively independent, so the first-order
        # inclusion-exclusion term is accurate to second order.
        used_fallback = True
        i_mp = fallback_estimate
        se = None
    else:
        i_mp = p_hat
        se = float(np.sqrt(p_hat * (1.0 - p_hat) / n_samples))

    return AtLeastOneMatchResult(
        i_mp=i_mp, se=se, n_samples=n_samples, loci_used=loci_used,
        candidates=[males[i].male_id for i in candidates],
        candidate_probs=candidate_probs,
        used_fallback=used_fallback, fallback_estimate=fallback_estimate,
    )
