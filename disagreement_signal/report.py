"""Render an evaluation dict as a Markdown report."""

from .metrics import mean


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    return "%.*f" % (digits, value)


def render_summary(results):
    pooled = results["pooled"]
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append(
        "Seed %d. Train %d rows; explained %d validation and %d test rows "
        "per model (3 models x 4 explainers)."
        % (
            results["seed"],
            results["n_train"],
            results["n_explained_val"],
            results["n_explained_test"],
        )
    )
    lines.append("")
    lines.append("## Pooled AUC for predicting model error (higher is better)")
    lines.append("")
    lines.append("| Signal | AUC |")
    lines.append("|---|---|")
    lines.append("| Explainer disagreement | %s |" % _fmt(pooled["auc_disagreement"]))
    lines.append("| Model confidence (1 - margin) | %s |" % _fmt(pooled["auc_confidence"]))
    lines.append("| Ensemble variance | %s |" % _fmt(pooled["auc_ensemble_variance"]))
    lines.append("| Disagreement + confidence (rank average) | %s |" % _fmt(pooled["auc_combined"]))
    lines.append("| Random | %s |" % _fmt(pooled["auc_random"]))
    lines.append("")
    lines.append("## Per model")
    lines.append("")
    lines.append(
        "| Model | Accuracy | AUC disagreement | AUC confidence | "
        "AUC ensemble var | AUC combined | Mean disagreement (errors) | "
        "Mean disagreement (correct) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in sorted(results["models"]):
        m = results["models"][name]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                name,
                _fmt(m["accuracy"]),
                _fmt(m["auc_disagreement"]),
                _fmt(m["auc_confidence"]),
                _fmt(m["auc_ensemble_variance"]),
                _fmt(m["auc_combined"]),
                _fmt(m["mean_disagreement_error"]),
                _fmt(m["mean_disagreement_correct"]),
            )
        )
    lines.append("")
    lines.append("## By planted region (pooled over models)")
    lines.append("")
    lines.append("| Region | Model-rows | Error rate | AUC disagreement | AUC confidence |")
    lines.append("|---|---|---|---|---|")
    for region in sorted(results["by_region"]):
        r = results["by_region"][region]
        lines.append(
            "| %s | %d | %s | %s | %s |"
            % (
                region,
                r["n_model_rows"],
                _fmt(r["error_rate"]),
                _fmt(r["auc_disagreement"]),
                _fmt(r["auc_confidence"]),
            )
        )
    lines.append("")
    lines.append("## Distrust-threshold triage (calibrated on validation, review target 20%)")
    lines.append("")
    lines.append(
        "| Model | Threshold | Review rate | Error capture | "
        "Error rate in queue | Error rate outside |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name in sorted(results["triage"]):
        t = results["triage"][name]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                name,
                _fmt(t["threshold"]),
                _fmt(t["review_rate"]),
                _fmt(t["error_capture"]),
                _fmt(t["flagged_error_rate"]),
                _fmt(t["unflagged_error_rate"]),
            )
        )
    lines.append("")
    captures = [results["triage"][m]["error_capture"] for m in results["triage"]]
    lines.append(
        "Mean error capture at a ~20%% review rate: %s of all errors are "
        "routed to review by the distrust threshold."
        % _fmt(mean(captures))
    )
    if "triage_combined" in results:
        combined_captures = [
            results["triage_combined"][m]["error_capture"]
            for m in results["triage_combined"]
        ]
        lines.append(
            "With the threshold calibrated on the combined "
            "disagreement + confidence score instead, mean error capture "
            "is %s." % _fmt(mean(combined_captures))
        )
    lines.append("")
    return "\n".join(lines)


def render_headline(results):
    pooled = results["pooled"]
    captures = [results["triage"][m]["error_capture"] for m in results["triage"]]
    lines = []
    lines.append("# Headline results")
    lines.append("")
    lines.append(
        "- Explainer disagreement predicts model error with pooled AUC %s, "
        "vs %s for model confidence, %s for ensemble variance, and %s for "
        "a random score (seed %d, synthetic data from a known process). "
        "The signal is model-class dependent: it is real for the decision "
        "tree, weak for k-NN, and absent for the linear model, whose "
        "explainers agree by construction."
        % (
            _fmt(pooled["auc_disagreement"]),
            _fmt(pooled["auc_confidence"]),
            _fmt(pooled["auc_ensemble_variance"]),
            _fmt(pooled["auc_random"]),
            results["seed"],
        )
    )
    lines.append(
        "- Rank-averaging disagreement with confidence gives pooled AUC "
        "%s, close to confidence alone (%s) and above disagreement "
        "alone: the two signals overlap heavily, and disagreement's "
        "independent contribution is small on this data."
        % (_fmt(pooled["auc_combined"]), _fmt(pooled["auc_confidence"]))
    )
    lines.append(
        "- A distrust threshold calibrated on validation data (20%% review "
        "target) captures %s of test errors in the review queue."
        % _fmt(mean(captures))
    )
    lines.append(
        "- All numbers are measured on seeded synthetic data with planted "
        "hard regions; they are not clinical or real-world results."
    )
    lines.append("")
    return "\n".join(lines)
