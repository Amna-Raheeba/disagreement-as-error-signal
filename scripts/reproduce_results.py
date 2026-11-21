"""Re-run the headline evaluation and fail if the numbers drift.

Compares a fresh run (seed 7) against the stored results/evaluation.json
within a small tolerance. Guards against silent changes to the data
process, models, explainers, or scoring.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disagreement_signal.evaluate import run_evaluation

TOLERANCE = 0.02
KEYS = ("auc_disagreement", "auc_confidence", "auc_ensemble_variance", "auc_combined", "auc_random")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    stored_path = os.path.join(here, "..", "results", "evaluation.json")
    with open(stored_path) as fh:
        stored = json.load(fh)
    fresh = run_evaluation(seed=stored["seed"])
    drift = []
    for key in KEYS:
        delta = abs(fresh["pooled"][key] - stored["pooled"][key])
        if delta > TOLERANCE:
            drift.append("%s drifted by %.4f" % (key, delta))
    if drift:
        print("HEADLINE DRIFT DETECTED:")
        for line in drift:
            print(" - %s" % line)
        return 1
    print("No drift: fresh pooled AUCs match stored results within %.2f." % TOLERANCE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
