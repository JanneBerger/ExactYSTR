"""Tests for ystr_pedigree.py.

Focus areas:
- compute_match_probs and compute_match_probs_fast must agree exactly
  (they are documented as numerically equivalent).
- Loading pedigrees from the example CSVs in Pedigrees/.
- Edge cases: untyped suspect, no untyped males, unsorted input.
"""

import glob
import os

import pytest

from ystr_pedigree import (
    Male,
    compute_match_probs,
    compute_match_probs_fast,
    load_pedigree_from_csv,
    make_stepwise_mutation_model,
)

PEDIGREE_DIR = os.path.join(os.path.dirname(__file__), "..", "Pedigrees")
PEDIGREE_FILES = sorted(glob.glob(os.path.join(PEDIGREE_DIR, "*.csv")))


@pytest.mark.parametrize("filepath", PEDIGREE_FILES)
@pytest.mark.parametrize("mu", [0.0, 0.1, 0.3])
def test_fast_matches_reference(filepath, mu):
    males, loci = load_pedigree_from_csv(filepath)
    for locus in loci:
        slow_probs, slow_avg = compute_match_probs(males, locus, mu)
        fast_probs, fast_avg = compute_match_probs_fast(males, locus, mu)

        assert slow_probs.keys() == fast_probs.keys()
        for i in slow_probs:
            assert slow_probs[i] == pytest.approx(fast_probs[i], abs=1e-9)
        assert slow_avg == pytest.approx(fast_avg, abs=1e-9)


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
    for fn in (compute_match_probs, compute_match_probs_fast):
        match_probs, average = fn(males, "M1", mu=0.1)
        assert match_probs == {}
        assert average == 0.0


def test_no_untyped_males_returns_empty():
    males = [
        Male(male_id=1, generation=0, father_id=-1, alleles={"M1": 13}),
        Male(male_id=2, generation=1, father_id=1, alleles={"M1": 13}),
    ]
    for fn in (compute_match_probs, compute_match_probs_fast):
        match_probs, average = fn(males, "M1", mu=0.1)
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
