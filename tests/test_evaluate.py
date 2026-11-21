import pytest

from disagreement_signal.evaluate import run_evaluation


@pytest.fixture(scope="module")
def results():
    return run_evaluation(seed=7, explain_limit=25)


def test_evaluation_structure(results):
    for key in ("seed", "models", "pooled", "by_region", "triage"):
        assert key in results
    assert set(results["models"].keys()) == {"logistic", "tree", "knn"}


def test_pooled_auc_bounds(results):
    for key in (
        "auc_disagreement",
        "auc_confidence",
        "auc_ensemble_variance",
        "auc_combined",
        "auc_random",
    ):
        assert 0.0 <= results["pooled"][key] <= 1.0


def test_random_auc_near_chance(results):
    assert 0.2 <= results["pooled"]["auc_random"] <= 0.8


def test_triage_review_rate_near_target(results):
    for stats in results["triage"].values():
        assert 0.05 <= stats["review_rate"] <= 0.45
        assert "threshold" in stats


def test_by_region_covers_regions(results):
    assert set(results["by_region"].keys()) <= {"normal", "interaction", "shift"}
    assert len(results["by_region"]) >= 1


def test_evaluation_deterministic_small():
    a = run_evaluation(seed=11, explain_limit=8)
    b = run_evaluation(seed=11, explain_limit=8)
    assert a["pooled"] == b["pooled"]
