"""Synthetic tabular data from a known process with planted hard regions.

The generating process is fully known, so ground truth (the label, the
region each row belongs to, and which features the process actually uses)
is exact. Three region types are planted on top of a mostly-linear base
process:

- ``normal``: the base linear process only.
- ``interaction``: rows inside a pocket where two features interact and
  partly cancel their main effects -- a linear model is systematically
  wrong here and explainers tend to split credit differently.
- ``shift``: rows inside a pocket where one feature's effect flips sign --
  the relationship the model learned from the majority no longer holds.

Labels are drawn from the true latent probability with a seeded RNG, so
the data is deterministic for a given seed.
"""

import math
import random

N_FEATURES = 8
FEATURE_NAMES = ["x%d" % i for i in range(N_FEATURES)]
# Features the true process uses (x5, x6, x7 are pure noise).
TRUE_FEATURES = [0, 1, 2, 3, 4]

REGIONS = ("normal", "interaction", "shift")


def latent_score(row):
    """True latent log-odds for a row, plus the region it belongs to."""
    x = row
    score = 1.2 * x[0] + 0.9 * x[1] - 0.8 * x[2] + 0.5 * x[3]
    region = "normal"
    # Interaction pocket: x0 and x1 both high -> their joint effect
    # cancels and reverses part of the main effects.
    if x[0] > 0.4 and x[1] > 0.4:
        score += -3.2 * x[0] * x[1] + 0.6
        region = "interaction"
    # Shift pocket: x2 in a band where its effect flips sign.
    if -0.2 < x[2] < 0.6 and x[4] > 0.3:
        score += 2.4 * x[2]
        if region == "normal":
            region = "shift"
    # Sharpen the log-odds so labels carry less irreducible noise: with
    # an unscaled score, most model "errors" are coin flips of the label
    # draw that no inference-time signal could predict. Scaling keeps the
    # same decision regions but concentrates genuine model failure in
    # the planted pockets.
    return score * 1.9, region


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def generate(n_rows, seed):
    """Generate n_rows rows. Returns (rows, labels, regions)."""
    rng = random.Random(seed)
    rows = []
    labels = []
    regions = []
    for _ in range(n_rows):
        row = [rng.gauss(0.0, 1.0) for _ in range(N_FEATURES)]
        score, region = latent_score(row)
        p = sigmoid(score)
        label = 1 if rng.random() < p else 0
        rows.append(row)
        labels.append(label)
        regions.append(region)
    return rows, labels, regions


def split_dataset(seed=7, n_train=800, n_val=300, n_test=400):
    """Deterministic train/val/test split from three seeded draws."""
    train = generate(n_train, seed)
    val = generate(n_val, seed + 1000)
    test = generate(n_test, seed + 2000)
    return train, val, test


def region_counts(regions):
    counts = {}
    for r in regions:
        counts[r] = counts.get(r, 0) + 1
    return counts
