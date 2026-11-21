"""Command line interface.

Usage:
    python -m disagreement_signal evaluate [--seed N] [--out DIR]
    python -m disagreement_signal demo [--seed N]
"""

import argparse
import csv
import json
import os
import sys

from . import data as data_mod
from . import models as models_mod
from .disagreement import disagreement_vector
from .evaluate import run_evaluation
from .explainers import explain_all
from .report import render_headline, render_summary


def _write_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "evaluation_summary.md"), "w") as fh:
        fh.write(render_summary(results))
    with open(os.path.join(out_dir, "HEADLINE.md"), "w") as fh:
        fh.write(render_headline(results))
    with open(os.path.join(out_dir, "evaluation.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "accuracy",
                "auc_disagreement",
                "auc_confidence",
                "auc_ensemble_variance",
                "auc_combined",
                "auc_random",
            ]
        )
        for name in sorted(results["models"]):
            m = results["models"][name]
            writer.writerow(
                [
                    name,
                    "%.4f" % m["accuracy"],
                    "%.4f" % m["auc_disagreement"],
                    "%.4f" % m["auc_confidence"],
                    "%.4f" % m["auc_ensemble_variance"],
                    "%.4f" % m["auc_combined"],
                    "%.4f" % m["auc_random"],
                ]
            )


def cmd_evaluate(args):
    results = run_evaluation(seed=args.seed, explain_limit=args.limit)
    _write_results(results, args.out)
    pooled = results["pooled"]
    print(
        "Pooled AUC: disagreement %.3f | confidence %.3f | ensemble %.3f "
        "| random %.3f"
        % (
            pooled["auc_disagreement"],
            pooled["auc_confidence"],
            pooled["auc_ensemble_variance"],
            pooled["auc_random"],
        )
    )
    print("Wrote results to %s" % args.out)
    return 0


def cmd_demo(args):
    train, _val, test = data_mod.split_dataset(seed=args.seed)
    train_rows, train_labels, _ = train
    test_rows, test_labels, test_regions = test
    models = models_mod.train_all(train_rows, train_labels)
    model = models["logistic"]
    for i in range(3):
        row = test_rows[i]
        attributions = explain_all(model, train_rows, row, args.seed + i)
        vec = disagreement_vector(attributions)
        pred = 1 if model.predict_proba_one(row) >= 0.5 else 0
        print(
            "Row %d (region %s): predicted %d, true %d, disagreement %.3f "
            "(rank spread %.3f, top-k gap %.3f, sign conflict %.3f)"
            % (
                i,
                test_regions[i],
                pred,
                test_labels[i],
                vec["disagreement_score"],
                vec["rank_spread"],
                vec["topk_gap"],
                vec["sign_conflict"],
            )
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="disagreement_signal")
    sub = parser.add_subparsers(dest="command")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--seed", type=int, default=7)
    p_eval.add_argument("--out", default="results")
    p_eval.add_argument("--limit", type=int, default=250)
    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "demo":
        return cmd_demo(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
