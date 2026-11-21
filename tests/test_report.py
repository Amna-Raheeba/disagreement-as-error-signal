from disagreement_signal.report import render_headline, render_summary


def _fake_results():
    return {
        "seed": 7,
        "n_train": 800,
        "n_explained_val": 250,
        "n_explained_test": 250,
        "pooled": {
            "auc_disagreement": 0.7,
            "auc_confidence": 0.6,
            "auc_ensemble_variance": 0.55,
            "auc_combined": 0.72,
            "auc_random": 0.5,
        },
        "models": {
            "logistic": {
                "accuracy": 0.8,
                "auc_disagreement": 0.7,
                "auc_confidence": 0.6,
                "auc_ensemble_variance": 0.55,
                "auc_combined": 0.72,
                "auc_random": 0.5,
                "mean_disagreement_error": 0.4,
                "mean_disagreement_correct": 0.2,
            }
        },
        "by_region": {
            "normal": {
                "n_model_rows": 300,
                "error_rate": 0.1,
                "auc_disagreement": 0.65,
                "auc_confidence": 0.6,
            }
        },
        "triage": {
            "logistic": {
                "threshold": 0.3,
                "review_rate": 0.2,
                "error_capture": 0.5,
                "flagged_error_rate": 0.4,
                "unflagged_error_rate": 0.1,
            }
        },
    }


def test_summary_contains_sections():
    text = render_summary(_fake_results())
    assert "Pooled AUC" in text
    assert "Per model" in text
    assert "Distrust-threshold" in text


def test_summary_contains_numbers():
    text = render_summary(_fake_results())
    assert "0.700" in text
    assert "logistic" in text


def test_headline_mentions_synthetic():
    text = render_headline(_fake_results())
    assert "synthetic" in text
    assert "0.700" in text
