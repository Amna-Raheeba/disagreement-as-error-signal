"""Demo: explain three test predictions and print their disagreement.

Run: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disagreement_signal import data as data_mod
from disagreement_signal import models as models_mod
from disagreement_signal.disagreement import disagreement_vector
from disagreement_signal.explainers import explain_all


def main():
    train, _val, test = data_mod.split_dataset(seed=7)
    train_rows, train_labels, _ = train
    test_rows, test_labels, test_regions = test
    models = models_mod.train_all(train_rows, train_labels)
    for name in ("logistic", "tree", "knn"):
        model = models[name]
        row = test_rows[0]
        attributions = explain_all(model, train_rows, row, 42)
        vec = disagreement_vector(attributions)
        pred = 1 if model.predict_proba_one(row) >= 0.5 else 0
        print(
            "%s: pred %d (true %d, region %s), disagreement %.3f"
            % (name, pred, test_labels[0], test_regions[0], vec["disagreement_score"])
        )
        for explainer in sorted(attributions):
            rounded = ["%.3f" % v for v in attributions[explainer]]
            print("  %-11s %s" % (explainer, ", ".join(rounded)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
