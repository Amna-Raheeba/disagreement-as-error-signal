import pytest

from disagreement_signal.disagreement import (
    disagreement_vector,
    ranks,
    spearman,
    topk_set,
)


def test_ranks_basic():
    assert ranks([10.0, 30.0, 20.0]) == [0.0, 2.0, 1.0]


def test_ranks_ties_average():
    r = ranks([1.0, 1.0, 2.0])
    assert r[0] == pytest.approx(0.5)
    assert r[1] == pytest.approx(0.5)
    assert r[2] == pytest.approx(2.0)


def test_spearman_identical():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_spearman_reversed():
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_constant_returns_one():
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 1.0


def test_topk_set():
    assert topk_set([0.1, -0.9, 0.5, 0.2], 2) == {1, 2}


def _identical():
    return {
        "a": [0.5, 0.3, -0.2, 0.1],
        "b": [0.5, 0.3, -0.2, 0.1],
        "c": [0.5, 0.3, -0.2, 0.1],
    }


def test_no_disagreement_when_identical():
    vec = disagreement_vector(_identical())
    assert vec["rank_spread"] == pytest.approx(0.0)
    assert vec["topk_gap"] == pytest.approx(0.0)
    assert vec["sign_conflict"] == pytest.approx(0.0)
    assert vec["disagreement_score"] == pytest.approx(0.0)


def test_sign_conflict_detected():
    attrs = {
        "a": [0.9, 0.1, 0.0, 0.0],
        "b": [-0.9, 0.1, 0.0, 0.0],
    }
    vec = disagreement_vector(attrs)
    assert vec["sign_conflict"] > 0.5


def test_reversed_rankings_high_spread():
    attrs = {
        "a": [0.4, 0.3, 0.2, 0.1],
        "b": [0.1, 0.2, 0.3, 0.4],
    }
    vec = disagreement_vector(attrs)
    assert vec["rank_spread"] == pytest.approx(1.0)


def test_score_in_unit_interval():
    attrs = {
        "a": [0.4, -0.3, 0.2, 0.1, 0.0],
        "b": [-0.1, 0.3, 0.2, -0.4, 0.5],
        "c": [0.0, 0.0, 0.9, -0.2, 0.1],
    }
    vec = disagreement_vector(attrs)
    assert 0.0 <= vec["disagreement_score"] <= 1.0


def test_vector_keys():
    vec = disagreement_vector(_identical())
    assert set(vec.keys()) == {
        "rank_spread",
        "topk_gap",
        "sign_conflict",
        "disagreement_score",
    }
