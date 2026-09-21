#!/usr/bin/env python3
"""
derive_multicopy_analytic.py
=============================
Independently derives closed-form match-probability formulas for
two-copy (c=2) analogues of the single-copy pedigrees A-C (Figure 1 /
Table 2), and cross-checks them against the shipped implementation
(`compute_match_probs_multicopy` in ystr_pedigree.py).

This is an *independent* re-implementation of the two-copy peeling
recursion (it does not import or call any function from
ystr_pedigree.py while deriving the formulas), so that agreement
between the two is a meaningful correctness check rather than a
tautology.

Key simplification used below: for c=2 copies, the message passed from
a child to its father is

    message = T1 @ beta_child @ T1.T

(a standard matrix congruence), because the joint two-copy mutation
kernel is the outer/tensor product of the single-copy kernel T1. This
avoids ever building the full (A^2 x A^2) transition matrix.

Pedigree design
----------------
A01_mc2, B01_mc, C01_mc, and C02_mc use a heterozygous suspect with
adjacent alleles, mirroring the single-copy A-C examples; A01_mc1 uses
a homozygous suspect instead. C01_mc's typed son carries the suspect's
own genotype; C02_mc's typed son carries a homozygous genotype --
unambiguous (unlike a heterozygous son, whose own copy-to-allele
mapping is itself unknown and would have to be marginalized over), and
reachable from the suspect's alleles by exactly one one-step mutation
per copy. mu (the code's all-copy rate) is chosen so that the per-copy,
unidirectional mutation rate equals z=0.05, matching Table 2
(mu_all = 1-(1-2z)^2). The symbolic derivation below uses its own
representative allele values (13,14 / 14,14 / 13,13); by the model's
translation invariance, the resulting formulas apply unchanged to the
different allele values actually used in the CSV files.

Usage
-----
    python derive_multicopy_analytic.py
"""

import os
import sympy as sp

from ystr_pedigree import load_pedigree_from_csv, compute_match_probs_multicopy

PEDIGREE_DIR = os.path.join(os.path.dirname(__file__), "Pedigrees")

# ---------------------------------------------------------------------------
# Independent symbolic two-copy peeling engine
# ---------------------------------------------------------------------------

z = sp.symbols('z', positive=True)

# Allele window: wide enough that probability mass starting near the
# centre can never reach the edge within <=3 meioses, so this exactly
# reproduces the "infinite line" formulas used throughout the paper
# (no boundary truncation).
_W = list(range(5, 23))
_idx = {a: i for i, a in enumerate(_W)}
_n = len(_W)


def _build_T1():
    T = sp.zeros(_n, _n)
    for a in _W:
        mass = 0
        for b in (a - 1, a + 1):
            if b in _idx:
                T[_idx[a], _idx[b]] = z
                mass += z
        T[_idx[a], _idx[a]] = 1 - mass
    return T


_T1 = _build_T1()
_T1t = _T1.T


def _emission_matrix(target):
    """target: None (untyped -> all ones), or an unordered pair (a,b)."""
    M = sp.zeros(_n, _n)
    if target is None:
        for i in range(_n):
            for j in range(_n):
                M[i, j] = 1
        return M
    a, b = target
    M[_idx[a], _idx[b]] = 1
    M[_idx[b], _idx[a]] = 1
    return M


def _depth(tree, i):
    d = 0
    while tree[i] is not None:
        i = tree[i]
        d += 1
    return d


def _compute_beta(tree, obs):
    """tree: dict child_id -> father_id (root has father_id None).
    obs: dict node_id -> None (untyped) or (a,b) target unordered pair.
    """
    children = {i: [] for i in tree}
    for c, f in tree.items():
        if f is not None:
            children[f].append(c)
    order = sorted(tree, key=lambda i: -_depth(tree, i))  # leaves first
    beta = {}
    for i in order:
        M = _emission_matrix(obs.get(i))
        for s in children[i]:
            msg = _T1 * beta[s] * _T1t
            M = M.multiply_elementwise(msg)
        beta[i] = M
    return beta


def _pedigree_prob(tree, obs):
    beta = _compute_beta(tree, obs)
    root = [i for i, f in tree.items() if f is None][0]
    return sum(beta[root][r, c] for r in range(_n) for c in range(_n))


def match_prob_formula(tree, obs, query_node, target):
    """Symbolic P(query_node's unordered genotype == target), in z."""
    P1 = _pedigree_prob(tree, obs)
    obs2 = dict(obs)
    obs2[query_node] = target
    P2 = _pedigree_prob(tree, obs2)
    return sp.nsimplify(sp.simplify(P2 / P1))


# ---------------------------------------------------------------------------
# The pedigrees, their trees/observations, and formula derivation
# ---------------------------------------------------------------------------

SUSPECT_SET = (13, 14)
HOMOZYGOUS = (13, 13)
MUTANT = (14, 14)

_TREE_CHAIN2 = {1: None, 2: 1}
_TREE_CHAIN3 = {1: None, 2: 1, 3: 2}

CASES = [
    # (csv file, male_id queried, tree, obs, target, description)
    ("A01_mc1.csv", 2, _TREE_CHAIN2,
     {1: HOMOZYGOUS, 2: None}, HOMOZYGOUS,
     "A (homozygous): suspect -> untyped son"),
    ("A01_mc2.csv", 2, _TREE_CHAIN2,
     {1: SUSPECT_SET, 2: None}, SUSPECT_SET,
     "A (heterozygous): suspect -> untyped son"),
    ("B01_mc.csv", 2, _TREE_CHAIN3,
     {1: SUSPECT_SET, 2: None, 3: None}, SUSPECT_SET,
     "B: suspect -> untyped -> untyped, query son"),
    ("B01_mc.csv", 3, _TREE_CHAIN3,
     {1: SUSPECT_SET, 2: None, 3: None}, SUSPECT_SET,
     "B: suspect -> untyped -> untyped, query grandson"),
    ("C01_mc.csv", 3, _TREE_CHAIN3,
     {1: SUSPECT_SET, 2: SUSPECT_SET, 3: None}, SUSPECT_SET,
     "C-same: suspect -> typed(=suspect) -> untyped"),
    ("C02_mc.csv", 3, _TREE_CHAIN3,
     {1: SUSPECT_SET, 2: MUTANT, 3: None}, SUSPECT_SET,
     "C-mutant: suspect -> typed(14,14) -> untyped"),
]

MU_ALL = 1.0 - (1.0 - 2 * 0.05) ** 2   # 0.19; per-copy unidirectional z=0.05


def main():
    print(f"{'file':11s} {'j':>2s}  {'formula (z)':<70s} {'z=0.05':>10s} "
          f"{'ES algorithm':>13s}  match")
    print("-" * 115)
    all_ok = True
    for fname, male_id, tree, obs, target, desc in CASES:
        formula = match_prob_formula(tree, obs, male_id, target)
        analytic = float(formula.subs(z, sp.Rational(1, 20)))

        males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, fname))
        id_to_idx = {m.male_id: i for i, m in enumerate(males)}
        mp, _ = compute_match_probs_multicopy(
            males, loci[0], n_copies=2, mu_all=MU_ALL,
        )
        es_value = mp[id_to_idx[male_id]]

        ok = abs(es_value - analytic) < 1e-6
        all_ok &= ok
        print(f"{fname:11s} {male_id:>2d}  {str(formula):<70s} "
              f"{analytic:>10.6f} {es_value:>13.6f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"    ({desc})")

    print("-" * 115)
    print("All pedigrees match the independently derived formula."
          if all_ok else "MISMATCHES FOUND -- see above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
