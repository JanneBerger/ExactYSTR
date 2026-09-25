"""Tests for ystr_pedigree.py.

Focus areas:
- Loading pedigrees from the example CSVs in Pedigrees/.
- Edge cases: untyped suspect, no untyped males, unsorted input.
- Multi-copy marker handling (DYS385-like loci).
"""

import glob
import os

import numpy as np
import pytest

from ystr_pedigree import (
    Male,
    compute_at_least_one_match,
    compute_match_probs_fast,
    compute_match_probs_multicopy,
    load_pedigree_from_csv,
    make_stepwise_mutation_model,
    sample_locus_matches,
)

PEDIGREE_DIR = os.path.join(os.path.dirname(__file__), "..", "Pedigrees")
PEDIGREE_FILES = sorted(glob.glob(os.path.join(PEDIGREE_DIR, "*.csv")))


# ---------------------------------------------------------------------------
# Single-copy tests (original)
# ---------------------------------------------------------------------------

def test_load_pedigree_from_csv_d01():
    males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "D01.csv"))

    assert loci == ["M1"]
    assert [m.male_id for m in males] == [1, 2, 3, 4]
    assert males[0].alleles == {"M1": 13}
    assert males[1].alleles == {"M1": 0}
    assert males[3].father_id == 3


def test_no_mutation_untyped_between_two_matching_typed_relatives():
    # D01: suspect (13) -> untyped -> untyped -> typed descendant (13).
    # With mu=0, the untyped males must carry the same allele as their
    # typed relatives, so the match probability is exactly 1.
    males, _ = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "D01.csv"))
    match_probs, average = compute_match_probs_fast(males, "M1", mu=0.0)

    assert match_probs.keys() == {1, 2}
    assert match_probs[1] == pytest.approx(1.0)
    assert match_probs[2] == pytest.approx(1.0)
    assert average == pytest.approx(1.0)


def test_no_mutation_untyped_between_two_mismatched_typed_relatives():
    # D02: suspect (13) -> untyped -> untyped -> typed descendant (14).
    # With mu=0, no allele change is possible, so the observed data has
    # zero likelihood, and no match probability can be computed.
    males, _ = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "D02.csv"))
    match_probs, average = compute_match_probs_fast(males, "M1", mu=0.0)

    assert match_probs == {}
    assert average == 0.0


def test_untyped_suspect_returns_empty():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 0}),
        Male(male_id=2, generation=1, father_id=1, alleles={"M1": 13}),
    ]
    match_probs, average = compute_match_probs_fast(males, "M1", mu=0.1)
    assert match_probs == {}
    assert average == 0.0


def test_no_untyped_males_returns_empty():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=1, alleles={"M1": 13}),
    ]
    match_probs, average = compute_match_probs_fast(males, "M1", mu=0.1)
    assert match_probs == {}
    assert average == 0.0


def test_fast_requires_sorted_pedigree():
    # Suspect (male_id=1, root) placed before its father (male_id=2) in
    # the list, so the father's index is not smaller than the child's.
    males = [
        Male(male_id=1, generation=0, father_id=2, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=-1, alleles={"M1": 0}),
    ]
    with pytest.raises(ValueError):
        compute_match_probs_fast(males, "M1", mu=0.1)


def test_stepwise_mutation_model_probabilities():
    p = make_stepwise_mutation_model(mu=0.2)

    assert p(10, 10) == pytest.approx(0.8)
    assert p(10, 9) == pytest.approx(0.1)
    assert p(10, 11) == pytest.approx(0.1)
    assert p(10, 12) == 0.0
    assert p(10, 8) == 0.0


# ---------------------------------------------------------------------------
# Multi-copy CSV loading
# ---------------------------------------------------------------------------

def test_load_multicopy_csv_a01_mc2():
    males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "A01_mc2.csv"))

    assert loci == ["M1"]
    assert len(males) == 2
    assert males[0].alleles == {"M1": (10, 11)}   # suspect, typed
    assert males[1].alleles == {"M1": 0}           # untyped (int 0)


# ---------------------------------------------------------------------------
# Multi-copy algorithm correctness
# ---------------------------------------------------------------------------

def _make_mc_chain(suspect_g, leaf_g, *, n_untyped=2):
    """Linear chain: suspect -> n_untyped untyped males -> typed leaf."""
    males = [Male(male_id=1, generation=0, father_id=-1,
                  alleles={"L": suspect_g})]
    for k in range(n_untyped):
        males.append(Male(male_id=k + 2, generation=k + 1,
                          father_id=k + 1, alleles={"L": 0}))
    males.append(Male(male_id=n_untyped + 2, generation=n_untyped + 1,
                      father_id=n_untyped + 1, alleles={"L": leaf_g}))
    return males


def test_multicopy_no_mutation_matching_leaf_mp_is_one():
    # Suspect {10,11} -> untyped -> untyped -> typed {10,11}.
    # mu=0 ⟹ no change possible ⟹ untyped males must carry {10,11} ⟹ MP=1.
    males = _make_mc_chain((10, 11), (10, 11))
    mp, avg = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.0)

    assert mp.keys() == {1, 2}
    assert mp[1] == pytest.approx(1.0)
    assert mp[2] == pytest.approx(1.0)
    assert avg == pytest.approx(1.0)


def test_multicopy_no_mutation_mismatched_leaf_returns_empty():
    # Suspect {10,11} -> ... -> typed {10,12}.
    # mu=0 ⟹ impossible path ⟹ P1=0 ⟹ empty result.
    males = _make_mc_chain((10, 11), (10, 12))
    mp, avg = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.0)

    assert mp == {}
    assert avg == 0.0


def test_multicopy_untyped_suspect_returns_empty():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": 0}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": (10, 11)}),
    ]
    mp, avg = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.1)
    assert mp == {}
    assert avg == 0.0


def test_multicopy_no_untyped_males_returns_empty():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (10, 11)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": (10, 11)}),
    ]
    mp, avg = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.1)
    assert mp == {}
    assert avg == 0.0


def test_multicopy_dispatch_via_fast():
    # compute_match_probs_fast should auto-dispatch to multicopy when alleles
    # are tuples and return the same result as the direct call.
    males = _make_mc_chain((10, 11), (10, 11))
    mp_fast, avg_fast = compute_match_probs_fast(males, "L", mu=0.0)
    mp_mc,   avg_mc   = compute_match_probs_multicopy(
        males, "L", n_copies=2, mu_all=0.0)

    assert mp_fast.keys() == mp_mc.keys()
    for k in mp_fast:
        assert mp_fast[k] == pytest.approx(mp_mc[k])
    assert avg_fast == pytest.approx(avg_mc)


def test_multicopy_dispatch_via_csv():
    # End-to-end: load A01_mc2.csv (suspect + 1 untyped) and run through normal API.
    males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "A01_mc2.csv"))
    mp, avg = compute_match_probs_fast(males, "M1", mu=0.0)

    assert mp.keys() == {1}         # index 1 is the one untyped male
    assert mp[1] == pytest.approx(1.0)


def test_multicopy_crossing_term_vs_shortcut():
    # Technical note Section 2.3: for a father with ordered state (11,12),
    # P(son reports {11,12}) = 0.81 (stay both) + 0.0025 (crossing) = 0.8125.
    # The split-into-two-single-markers shortcut returns 0.8100 and misses
    # the crossing term.  Here we verify the exact engine returns > 0.81.
    #
    # With per-copy mu=0.1: T(11->11)=T(12->12)=0.9, T(11->12)=T(12->11)=0.05
    # mu_all for a 2-copy marker with per-copy mu=0.1: mu_all = 1-(1-0.1)^2 = 0.19
    #
    # P(son={11,12} | father ordered (11,12)) = 0.9*0.9 + 0.05*0.05 = 0.8125
    # Averaging over both orderings of father's state: MP = 0.8125
    mu_all = 1.0 - (1.0 - 0.1) ** 2   # 0.19 → per-copy mu = 0.1
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (11, 12)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mp, _ = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=mu_all)

    # Must exceed the shortcut value of 0.8100 and equal the exact value 0.8125
    assert mp[1] > 0.810
    assert mp[1] == pytest.approx(0.8125, abs=1e-6)


def test_multicopy_homozygous_suspect():
    # Suspect {11,11} -> untyped son.  mu=0: son must also carry {11,11} -> MP=1.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (11, 11)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mp, avg = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.0)
    assert mp[1] == pytest.approx(1.0)


def test_multicopy_requires_sorted_pedigree():
    males = [
        Male(male_id=1, generation=0, father_id=2, alleles={"L": (10, 11)}),
        Male(male_id=2, generation=1, father_id=-1, alleles={"L": 0}),
    ]
    with pytest.raises(ValueError):
        compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.1)


def test_multicopy_symmetric_siblings():
    # Two untyped sons of the suspect must receive identical MP (structural
    # symmetry).  Any asymmetry would indicate a bug in the outside pass.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (10, 11)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
        Male(male_id=3, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mp, _ = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=0.1)
    assert mp[1] == pytest.approx(mp[2], rel=1e-9)


def test_multicopy_per_copy_rate_conversion():
    # For a 2-copy marker with mu_all=0.19 (= 1 - 0.9^2):
    #   per-copy mu = 1 - sqrt(1 - 0.19) = 1 - 0.9 = 0.1
    # For a homozygous suspect {12,12} → untyped son:
    #   MP = P(both copies stay at 12) = T(12->12)^2 = (1-0.1)^2 = 0.81
    mu_all = 1.0 - (1.0 - 0.1) ** 2   # = 0.19
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (12, 12)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mp, _ = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=mu_all)

    # Each copy must stay at 12: (1-mu_per_copy)^2 = 0.9^2 = 0.81
    assert mp[1] == pytest.approx(0.81, abs=1e-6)


# ---------------------------------------------------------------------------
# Multi-copy analogues of pedigrees A-C, verified against independently
# derived closed-form formulas (see derive_multicopy_analytic.py).
#
# A01_mc2/B01_mc/C01_mc/C02_mc use a heterozygous suspect with adjacent
# alleles (10,11 or 13,14), mirroring the single-copy A-C examples
# (Figure 1 / Table 2); A01_mc1 uses a homozygous suspect instead. mu
# (the code's all-copy rate) is chosen so that the per-copy,
# unidirectional mutation rate equals z=0.05 (i.e. mu_all =
# 1-(1-2z)^2 = 0.19), matching the z=0.05 used throughout Table 2.
#
# C01_mc's typed son carries the suspect's own genotype (13,14); C02_mc's
# typed son carries the homozygous genotype (14,14) -- unambiguous
# (unlike a heterozygous son, whose own internal copy-to-allele mapping
# is itself unknown and must be marginalized over), and reachable from
# the suspect's alleles by exactly one one-step mutation per copy.
# ---------------------------------------------------------------------------

MC_MU_ALL = 1.0 - (1.0 - 2 * 0.05) ** 2   # 0.19; per-copy unidirectional z=0.05


@pytest.mark.parametrize("fname,male_id,expected", [
    # Pedigree A analogue: suspect -> 1 untyped son.
    ("A01_mc1.csv", 2, 0.81),       # homozygous suspect (10,10)
    ("A01_mc2.csv", 2, 0.8125),     # heterozygous suspect (10,11)
    # Pedigree B analogue: suspect -> untyped -> untyped.
    ("B01_mc.csv", 2, 0.8125),
    ("B01_mc.csv", 3, 0.672325),
    # Pedigree C analogue: suspect -> typed son -> untyped grandson.
    ("C01_mc.csv", 3, 0.8125),      # son typed identical to suspect
    ("C02_mc.csv", 3, 0.09),        # son typed homozygous mutant (14,14)
])
def test_multicopy_AC_analogues_match_analytic_formula(fname, male_id, expected):
    males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, fname))
    id_to_idx = {m.male_id: i for i, m in enumerate(males)}
    mp, _ = compute_match_probs_multicopy(
        males, loci[0], n_copies=2, mu_all=MC_MU_ALL,
    )
    assert mp[id_to_idx[male_id]] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# "At least one match" (i-MP): exact sampling from the peeling messages
# ---------------------------------------------------------------------------

def test_sample_locus_matches_matches_analytic_single_copy():
    # D01: suspect (13) -> two untyped -> typed descendant (13), mu=0.1.
    # The sampled match rate for each untyped male must converge to the
    # exact analytic match probability (law of large numbers).
    males, _ = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "D01.csv"))
    mp, _ = compute_match_probs_fast(males, "M1", mu=0.1)

    rng = np.random.default_rng(0)
    matches, excluded, usable = sample_locus_matches(
        males, "M1", mu_all=0.1, n_samples=100_000, rng=rng
    )
    assert usable
    assert not excluded.any()
    for i, p in mp.items():
        assert matches[:, i].mean() == pytest.approx(p, abs=0.01)


def test_sample_locus_matches_multicopy_matches_analytic():
    # Section 2.3 crossing-term example: father (11,12), mu_pc=0.1 -> exact
    # MP = 0.8125.  The sampled rate must converge to the same value.
    mu_all = 1.0 - (1.0 - 0.1) ** 2
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (11, 12)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mp, _ = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=mu_all)

    rng = np.random.default_rng(1)
    matches, excluded, usable = sample_locus_matches(
        males, "L", mu_all=mu_all, n_copies=2, n_samples=100_000, rng=rng
    )
    assert usable
    assert not excluded.any()
    assert matches[:, 1].mean() == pytest.approx(mp[1], abs=0.01)


def test_sample_locus_matches_flags_typed_mismatch_as_excluded():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"M1": 14}),  # 1-step mismatch
    ]
    matches, excluded, usable = sample_locus_matches(
        males, "M1", mu_all=0.1, n_samples=100
    )
    assert usable
    assert excluded.tolist() == [False, True]


def test_at_least_one_match_two_independent_sons_union_formula():
    # Two untyped sons of a typed suspect, single locus. Given the root,
    # their alleles are conditionally independent, so
    # i-MP = 1 - (1-p)^2 where p is each son's individual match probability.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"M1": 0}),
        Male(male_id=3, generation=1, father_id=1,  alleles={"M1": 0}),
    ]
    mp, _ = compute_match_probs_fast(males, "M1", mu=0.1)
    p = mp[1]
    expected = 1.0 - (1.0 - p) ** 2

    res = compute_at_least_one_match(males, ["M1"], mu=0.1, n_samples=100_000, seed=0)
    assert not res.used_fallback
    assert res.i_mp == pytest.approx(expected, abs=0.01)
    assert res.se is not None and res.se > 0


def test_at_least_one_match_typed_matching_descendant_is_not_a_candidate():
    # D01: individual 4 is typed and already matches the suspect, but
    # i-MP is defined over untyped candidates only, so he does not make
    # i-MP trivially 1 -- only the two untyped individuals (2, 3) count,
    # and i-MP is their real "at least one matches" probability.
    males, _ = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "D01.csv"))
    res = compute_at_least_one_match(males, ["M1"], mu=0.1, n_samples=200_000, seed=0)
    assert res.candidates == [2, 3]
    assert res.i_mp == pytest.approx(0.993939, abs=0.01)
    assert res.se is not None and res.se > 0


def test_at_least_one_match_no_untyped_males_is_zero():
    # Both males are typed (and match); with no untyped candidates,
    # i-MP is exactly 0 -- a match already present in the typed data is
    # not what i-MP measures.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"M1": 13}),
    ]
    res = compute_at_least_one_match(males, ["M1"], mu=0.1, n_samples=100, seed=0)
    assert res.i_mp == 0.0
    assert res.se == 0.0
    assert res.candidates == []


def test_at_least_one_match_excludes_typed_mismatch_at_other_locus():
    # Male 2 is untyped at A but typed-and-mismatching at B (1-step off),
    # so he can never match at both loci simultaneously and must be
    # excluded from the candidate set entirely.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"A": 13, "B": 20}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"A": 0,  "B": 21}),
    ]
    res = compute_at_least_one_match(males, ["A", "B"], mu=0.1, n_samples=5000, seed=0)
    assert res.candidates == []
    assert res.i_mp == 0.0


def test_at_least_one_match_fallback_matches_first_order_sum():
    # A single untyped son, but many loci each with a small per-locus
    # match probability (mu=0.7 -> P(no mutation)=0.3 per locus), so the
    # combined probability (0.3**10 ~ 5.9e-6) is far below the sampling
    # resolution of n_samples=1000 -- the estimator must fall back to the
    # first-order inclusion-exclusion sum (here just P_i, one candidate).
    loci = [f"L{k}" for k in range(10)]
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={loc: 13 for loc in loci}),
        Male(male_id=2, generation=1, father_id=1,  alleles={loc: 0 for loc in loci}),
    ]
    res = compute_at_least_one_match(males, loci, mu=0.7, n_samples=1000, seed=0)
    assert res.used_fallback
    assert res.se is None
    expected = 0.3 ** 10
    assert res.i_mp == pytest.approx(expected, rel=1e-9)
    assert res.candidate_probs[2] == pytest.approx(expected, rel=1e-9)


def test_at_least_one_match_no_usable_loci_returns_zero():
    # Suspect untyped at the only locus -> nothing can be computed.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 0}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"M1": 13}),
    ]
    res = compute_at_least_one_match(males, ["M1"], mu=0.1, n_samples=100, seed=0)
    assert res.loci_used == []
    assert res.candidates == []
    assert res.i_mp == 0.0


def test_at_least_one_match_reproducible_with_seed():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"M1": 0}),
        Male(male_id=3, generation=1, father_id=1,  alleles={"M1": 0}),
    ]
    res_a = compute_at_least_one_match(males, ["M1"], mu=0.1, n_samples=5000, seed=123)
    res_b = compute_at_least_one_match(males, ["M1"], mu=0.1, n_samples=5000, seed=123)
    assert res_a.i_mp == res_b.i_mp


# ---------------------------------------------------------------------------
# Mutation-rate convention: mu is the overall per-locus (all-copy) rate
# at every public entry point.
# ---------------------------------------------------------------------------

def test_at_least_one_match_multicopy_uses_all_copy_rate():
    # One untyped son of a {11,12} suspect: i-MP is just his MP, which
    # must equal compute_match_probs_multicopy at the same mu_all.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (11, 12)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mu_all = 0.19
    mp, _ = compute_match_probs_multicopy(males, "L", n_copies=2, mu_all=mu_all)
    res = compute_at_least_one_match(males, ["L"], mu=mu_all, n_samples=200_000, seed=1)
    assert res.candidate_probs[2] == pytest.approx(mp[1], rel=1e-12)
    assert abs(res.i_mp - mp[1]) < 5 * res.se


def test_multicopy_two_step_fraction_preserved_per_copy():
    # Homozygous {12,12} suspect -> untyped son: MP = (1 - mu_pc)^2, where
    # mu_pc = 1 - (1 - mu_all)^(1/c) is the per-copy *total* rate,
    # independent of how it is split into one- and two-step mutations.
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"L": (12, 12)}),
        Male(male_id=2, generation=1, father_id=1,  alleles={"L": 0}),
    ]
    mu_all, f = 0.19, 0.3
    mp, _ = compute_match_probs_multicopy(
        males, "L", n_copies=2, mu_all=mu_all * (1 - f), mu2_all=mu_all * f,
    )
    assert mp[1] == pytest.approx(0.81, rel=1e-12)
