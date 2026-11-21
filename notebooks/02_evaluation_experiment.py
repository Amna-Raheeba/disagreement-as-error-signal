"""Evaluation experiment: disagreement vs confidence vs ensemble variance.

Runs the full headline experiment on a reduced explain limit so the
notebook finishes quickly, and prints the pooled AUC table.

Run: python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disagreement_signal.evaluate import run_evaluation


def main():
    results = run_evaluation(seed=7, explain_limit=60)
    pooled = results["pooled"]
    print("Pooled AUC for predicting model error (reduced run, 60 rows/model):")
    print("  disagreement       %.3f" % pooled["auc_disagreement"])
    print("  confidence         %.3f" % pooled["auc_confidence"])
    print("  ensemble variance  %.3f" % pooled["auc_ensemble_variance"])
    print("  random             %.3f" % pooled["auc_random"])
    for region in sorted(results["by_region"]):
        r = results["by_region"][region]
        print(
            "  region %-11s error rate %.3f, disagreement AUC %s"
            % (
                region,
                r["error_rate"],
                "%.3f" % r["auc_disagreement"] if r["auc_disagreement"] is not None else "n/a",
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
