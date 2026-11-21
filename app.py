"""Streamlit dashboard for disagreement-as-error-signal.

Three tabs:
1. Explain a prediction -- run all four explainers on one test row and
   show the attributions and the disagreement vector.
2. Distrust threshold -- calibrate a threshold on validation disagreement
   scores and inspect the review queue it produces on test rows.
3. Evaluation -- the stored headline results (results/evaluation.json),
   with a pooled-AUC chart.

Run: streamlit run app.py
"""

import json
import os
import random

import streamlit as st

from disagreement_signal import data as data_mod
from disagreement_signal import models as models_mod
from disagreement_signal.disagreement import disagreement_vector
from disagreement_signal.explainers import explain_all
from disagreement_signal.metrics import calibrate_threshold, triage_stats


@st.cache_resource
def load_world():
    train, val, test = data_mod.split_dataset(seed=7)
    models = models_mod.train_all(train[0], train[1])
    return train, val, test, models


def tab_explain():
    st.header("Explain a prediction")
    train, _val, test, models = load_world()
    test_rows, test_labels, test_regions = test
    model_name = st.selectbox("Model", sorted(models.keys()))
    idx = st.slider("Test row", 0, min(49, len(test_rows) - 1), 0)
    row = test_rows[idx]
    model = models[model_name]
    attributions = explain_all(model, train[0], row, 1000 + idx)
    vec = disagreement_vector(attributions)
    proba = model.predict_proba_one(row)
    st.write(
        "Region: %s | Predicted class %d (p=%.3f) | True label %d"
        % (test_regions[idx], 1 if proba >= 0.5 else 0, proba, test_labels[idx])
    )
    st.write("**Disagreement score: %.3f**" % vec["disagreement_score"])
    st.write(
        "Rank spread %.3f | Top-k gap %.3f | Sign conflict %.3f"
        % (vec["rank_spread"], vec["topk_gap"], vec["sign_conflict"])
    )
    chart_data = {}
    for name in sorted(attributions):
        chart_data[name] = attributions[name]
    st.bar_chart(chart_data)


def _scores_for(model, background, rows, labels, seed_base):
    scores = []
    errors = []
    for i, row in enumerate(rows):
        attributions = explain_all(model, background, row, seed_base + i)
        vec = disagreement_vector(attributions)
        scores.append(vec["disagreement_score"])
        pred = 1 if model.predict_proba_one(row) >= 0.5 else 0
        errors.append(pred != labels[i])
    return scores, errors


def tab_threshold():
    st.header("Distrust threshold")
    train, val, test, models = load_world()
    model_name = st.selectbox("Model", sorted(models.keys()), key="thr_model")
    model = models[model_name]
    n = 40
    val_scores, _ = _scores_for(model, train[0], val[0][:n], val[1][:n], 5000)
    test_scores, test_errors = _scores_for(
        model, train[0], test[0][:n], test[1][:n], 9000
    )
    target = st.slider("Target review rate", 0.05, 0.5, 0.2, step=0.05)
    threshold = calibrate_threshold(val_scores, [False] * len(val_scores), target)
    stats = triage_stats(test_scores, test_errors, threshold)
    st.write("Calibrated threshold: %.3f" % threshold)
    st.write(
        "Review rate %.2f | Error capture %.2f | Error rate in queue %.2f "
        "| Error rate outside %.2f"
        % (
            stats["review_rate"],
            stats["error_capture"],
            stats["flagged_error_rate"],
            stats["unflagged_error_rate"],
        )
    )
    st.bar_chart({"disagreement": test_scores})


def tab_evaluation():
    st.header("Evaluation")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "evaluation.json"
    )
    if not os.path.exists(path):
        st.write("No stored results yet. Run: python -m disagreement_signal evaluate")
        return
    with open(path) as fh:
        results = json.load(fh)
    pooled = results["pooled"]
    st.write("Pooled AUC for predicting model error:")
    chart = {
        "disagreement": pooled["auc_disagreement"],
        "confidence": pooled["auc_confidence"],
        "ensemble variance": pooled["auc_ensemble_variance"],
        "combined": pooled["auc_combined"],
        "random": pooled["auc_random"],
    }
    st.bar_chart(chart)
    st.write(
        "Disagreement AUC %.3f vs confidence %.3f (seed %d, synthetic data)."
        % (pooled["auc_disagreement"], pooled["auc_confidence"], results["seed"])
    )
    rng = random.Random(0)
    st.write("Random check value: %.3f" % rng.random())


def main():
    st.title("Disagreement as an Error Signal")
    st.write(
        "When post-hoc explainers disagree about a prediction, the model "
        "is more likely to be wrong. This dashboard makes that signal "
        "inspectable: no labels are used at inference time."
    )
    tab1, tab2, tab3 = st.tabs(["Explain", "Distrust threshold", "Evaluation"])
    with tab1:
        tab_explain()
    with tab2:
        tab_threshold()
    with tab3:
        tab_evaluation()


if __name__ == "__main__":
    main()
