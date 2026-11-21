"""One explained prediction, as JSON, for the examples/run_demo.py script."""

import json
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
    model = models["tree"]
    row = test_rows[1]
    attributions = explain_all(model, train_rows, row, 123)
    vec = disagreement_vector(attributions)
    payload = {
        "row": ["%.3f" % v for v in row],
        "region": test_regions[1],
        "true_label": test_labels[1],
        "predicted_proba": model.predict_proba_one(row),
        "attributions": {
            name: ["%.4f" % v for v in vec_attr]
            for name, vec_attr in sorted(attributions.items())
        },
        "disagreement": {
            key: round(value, 4) for key, value in sorted(vec.items())
        },
    }
    out = os.path.join(os.path.dirname(__file__), "demo_case.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("Wrote %s" % out)
    print(json.dumps(payload["disagreement"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
