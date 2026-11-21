"""The headline experiment.

Question: does explainer disagreement -- computable at inference time,
with no labels -- predict whether the model is wrong, better than the
standard label-free signals (model confidence and ensemble variance)?

Design (seeded, deterministic):
- Data from ``data.split_dataset`` (known process, planted hard regions).
- Three stdlib models are trained. Each model is explained by all four
  explainers on validation and test rows.
- Per model: AUC of the composite disagreement score for predicting that
  model's errors, vs AUC of (a) low confidence, (b) ensemble variance
  across the three models, (c) a seeded random score.
- AUCs are also broken down by planted region (normal / interaction /
  shift), pooling all models.
- A distrust threshold is calibrated on validation disagreement scores
  (target review rate 20%) and applied to the test set: review rate,
  error capture, and the error rate inside vs outside the review queue.
"""

import random

from . import data as data_mod
from . import models as models_mod
from .disagreement import disagreement_vector
from .explainers import explain_all
from .metrics import auc, calibrate_threshold, mean, rank_normalise, triage_stats

EXPLAIN_LIMIT = 250


def _score_rows(models, background, rows, labels, seed_base):
    """Explain every (model, row) pair; return per-model record lists."""
    per_model = {}
    probs = {}
    for name, model in models.items():
        probs[name] = [model.predict_proba_one(r) for r in rows]
    for name, model in models.items():
        records = []
        for i, row in enumerate(rows):
            attributions = explain_all(model, background, row, seed_base + i)
            vec = disagreement_vector(attributions)
            pred = 1 if probs[name][i] >= 0.5 else 0
            all_probs = [probs[m][i] for m in models]
            p_mean = mean(all_probs)
            ens_var = mean([(p - p_mean) ** 2 for p in all_probs])
            records.append(
                {
                    "disagreement": vec["disagreement_score"],
                    "rank_spread": vec["rank_spread"],
                    "topk_gap": vec["topk_gap"],
                    "sign_conflict": vec["sign_conflict"],
                    "confidence_uncertainty": 1.0 - abs(2.0 * probs[name][i] - 1.0),
                    "ensemble_variance": ens_var,
                    "error": pred != labels[i],
                }
            )
        per_model[name] = records
    return per_model


def run_evaluation(seed=7, explain_limit=EXPLAIN_LIMIT):
    train, val, test = data_mod.split_dataset(seed=seed)
    train_rows, train_labels, _ = train
    val_rows, val_labels, val_regions = val
    test_rows, test_labels, test_regions = test
    models = models_mod.train_all(train_rows, train_labels)
    background = train_rows

    val_rows = val_rows[:explain_limit]
    val_labels = val_labels[:explain_limit]
    val_regions = val_regions[:explain_limit]
    test_rows = test_rows[:explain_limit]
    test_labels = test_labels[:explain_limit]
    test_regions = test_regions[:explain_limit]

    val_scored = _score_rows(models, background, val_rows, val_labels, seed * 10)
    test_scored = _score_rows(
        models, background, test_rows, test_labels, seed * 10 + 500000
    )

    rng = random.Random(seed + 99)
    random_scores = [rng.random() for _ in test_rows]

    # Combined signal: rank-average of disagreement and confidence
    # uncertainty. Computed per model on each split, so no label or
    # test information leaks into the ranking.
    for scored in (val_scored, test_scored):
        for name in models:
            recs = scored[name]
            d_rank = rank_normalise([r["disagreement"] for r in recs])
            c_rank = rank_normalise(
                [r["confidence_uncertainty"] for r in recs]
            )
            for i, rec in enumerate(recs):
                rec["combined"] = 0.5 * (d_rank[i] + c_rank[i])

    per_model = {}
    for name in models:
        recs = test_scored[name]
        errors = [r["error"] for r in recs]
        per_model[name] = {
            "accuracy": 1.0 - mean([1.0 if e else 0.0 for e in errors]),
            "error_rate": mean([1.0 if e else 0.0 for e in errors]),
            "auc_disagreement": auc([r["disagreement"] for r in recs], errors),
            "auc_confidence": auc(
                [r["confidence_uncertainty"] for r in recs], errors
            ),
            "auc_ensemble_variance": auc(
                [r["ensemble_variance"] for r in recs], errors
            ),
            "auc_combined": auc([r["combined"] for r in recs], errors),
            "auc_random": auc(random_scores, errors),
            "mean_disagreement_error": mean(
                [r["disagreement"] for r in recs if r["error"]]
            ),
            "mean_disagreement_correct": mean(
                [r["disagreement"] for r in recs if not r["error"]]
            ),
        }

    pooled = {
        "auc_disagreement": mean(
            [per_model[m]["auc_disagreement"] for m in per_model]
        ),
        "auc_confidence": mean(
            [per_model[m]["auc_confidence"] for m in per_model]
        ),
        "auc_ensemble_variance": mean(
            [per_model[m]["auc_ensemble_variance"] for m in per_model]
        ),
        "auc_random": mean([per_model[m]["auc_random"] for m in per_model]),
        "auc_combined": mean(
            [per_model[m]["auc_combined"] for m in per_model]
        ),
    }

    by_region = {}
    for region in data_mod.REGIONS:
        d_scores = []
        c_scores = []
        errs = []
        for name in models:
            recs = test_scored[name]
            for i, rec in enumerate(recs):
                if test_regions[i] == region:
                    d_scores.append(rec["disagreement"])
                    c_scores.append(rec["confidence_uncertainty"])
                    errs.append(rec["error"])
        if errs and any(errs) and not all(errs):
            by_region[region] = {
                "n_model_rows": len(errs),
                "error_rate": mean([1.0 if e else 0.0 for e in errs]),
                "auc_disagreement": auc(d_scores, errs),
                "auc_confidence": auc(c_scores, errs),
            }
        elif errs:
            by_region[region] = {
                "n_model_rows": len(errs),
                "error_rate": mean([1.0 if e else 0.0 for e in errs]),
                "auc_disagreement": None,
                "auc_confidence": None,
            }

    # Distrust threshold: calibrate per model on validation, apply to test.
    triage = {}
    for name in models:
        threshold = calibrate_threshold(
            [r["disagreement"] for r in val_scored[name]],
            [r["error"] for r in val_scored[name]],
            target_review_rate=0.20,
        )
        stats = triage_stats(
            [r["disagreement"] for r in test_scored[name]],
            [r["error"] for r in test_scored[name]],
            threshold,
        )
        stats["threshold"] = threshold
        triage[name] = stats

    # Same triage, but the threshold is calibrated on the combined score.
    triage_combined = {}
    for name in models:
        threshold = calibrate_threshold(
            [r["combined"] for r in val_scored[name]],
            [r["error"] for r in val_scored[name]],
            target_review_rate=0.20,
        )
        stats = triage_stats(
            [r["combined"] for r in test_scored[name]],
            [r["error"] for r in test_scored[name]],
            threshold,
        )
        stats["threshold"] = threshold
        triage_combined[name] = stats

    return {
        "seed": seed,
        "n_train": len(train_rows),
        "n_explained_val": len(val_rows),
        "n_explained_test": len(test_rows),
        "models": per_model,
        "pooled": pooled,
        "by_region": by_region,
        "triage": triage,
        "triage_combined": triage_combined,
        "test_region_counts": data_mod.region_counts(test_regions),
    }
