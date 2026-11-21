import pytest

from disagreement_signal.metrics import (
    auc,
    calibrate_threshold,
    mean,
    rank_normalise,
    triage_stats,
)


def test_auc_perfect():
    assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == pytest.approx(1.0)


def test_auc_worst():
    assert auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == pytest.approx(0.0)


def test_auc_chance_for_constant():
    assert auc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == pytest.approx(0.5)


def test_auc_no_positives():
    assert auc([0.1, 0.2], [False, False]) == 0.5


def test_auc_ties():
    # One positive tied with one negative, one positive above.
    value = auc([0.5, 0.5, 0.9], [True, False, True])
    assert value == pytest.approx(0.75)


def test_calibrate_threshold_review_rate():
    scores = [i / 100.0 for i in range(101)]
    thr = calibrate_threshold(scores, [False] * 101, target_review_rate=0.20)
    flagged = [s for s in scores if s >= thr]
    assert 0.15 <= len(flagged) / 101.0 <= 0.25


def test_calibrate_threshold_empty():
    assert calibrate_threshold([], []) == 0.0


def test_triage_stats_perfect_capture():
    scores = [0.9, 0.8, 0.1, 0.2]
    errors = [True, True, False, False]
    stats = triage_stats(scores, errors, 0.5)
    assert stats["error_capture"] == pytest.approx(1.0)
    assert stats["review_rate"] == pytest.approx(0.5)
    assert stats["flagged_error_rate"] == pytest.approx(1.0)
    assert stats["unflagged_error_rate"] == pytest.approx(0.0)


def test_triage_stats_no_flagged():
    stats = triage_stats([0.1, 0.2], [True, False], 0.99)
    assert stats["n_flagged"] == 0
    assert stats["error_capture"] == pytest.approx(0.0)
    assert stats["unflagged_error_rate"] == pytest.approx(0.5)


def test_triage_stats_keys():
    stats = triage_stats([0.1], [True], 0.05)
    for key in (
        "review_rate",
        "error_capture",
        "flagged_error_rate",
        "unflagged_error_rate",
        "overall_error_rate",
    ):
        assert key in stats


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert mean([]) == 0.0


def test_rank_normalise_order():
    out = rank_normalise([10.0, 30.0, 20.0])
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)


def test_rank_normalise_ties_and_single():
    out = rank_normalise([5.0, 5.0])
    assert out[0] == pytest.approx(0.5)
    assert rank_normalise([1.0]) == [0.5]
