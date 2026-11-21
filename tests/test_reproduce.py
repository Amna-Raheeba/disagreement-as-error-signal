import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_stored_results_exist_and_parse():
    path = os.path.join(ROOT, "results", "evaluation.json")
    assert os.path.exists(path)
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["seed"] == 7
    for key in ("auc_disagreement", "auc_confidence", "auc_ensemble_variance", "auc_combined", "auc_random"):
        assert key in payload["pooled"]


def test_stored_headline_matches_json():
    with open(os.path.join(ROOT, "results", "evaluation.json")) as fh:
        payload = json.load(fh)
    with open(os.path.join(ROOT, "results", "HEADLINE.md")) as fh:
        headline = fh.read()
    assert "%.3f" % payload["pooled"]["auc_disagreement"] in headline


def test_reproduce_script_present():
    path = os.path.join(ROOT, "scripts", "reproduce_results.py")
    assert os.path.exists(path)
    with open(path) as fh:
        text = fh.read()
    assert "TOLERANCE" in text
    assert "run_evaluation" in text
