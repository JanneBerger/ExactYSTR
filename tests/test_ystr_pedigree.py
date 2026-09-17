"""Tests for ystr_pedigree.py.

Focus areas:
- Loading pedigrees from the example CSVs in Pedigrees/.
- Edge cases: untyped suspect, no untyped males, unsorted input.
- Multi-copy marker handling (DYS385-like loci).
"""

import glob
import os

import pytest

from ystr_pedigree import (
    Male,
    compute_match_probs_fast,
    compute_match_probs_multicopy,
    load_pedigree_from_csv,
    make_stepwise_mutation_model,
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

def test_load_multicopy_csv_mc01():
    males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "MC01.csv"))

    assert loci == ["DYS385"]
    assert len(males) == 2
    assert males[0].alleles == {"DYS385": (10, 11)}   # suspect, typed
    assert males[1].alleles == {"DYS385": 0}           # untyped (int 0)


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
    # End-to-end: load MC01.csv (suspect + 1 untyped) and run through normal API.
    males, loci = load_pedigree_from_csv(os.path.join(PEDIGREE_DIR, "MC01.csv"))
    mp, avg = compute_match_probs_fast(males, "DYS385", mu=0.0)

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
