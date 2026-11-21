"""Four stdlib post-hoc explainers.

Each explainer takes a model, a background dataset (for reference values),
and one row, and returns a signed attribution vector of length n_features:
positive = pushes the prediction toward class 1, negative = toward 0.

- ``loco``: leave-one-feature-out -- replace each feature with its
  background mean and measure the probability change.
- ``permutation``: per-instance permutation -- shuffle one feature's
  values across background rows (keeping this row's other features) and
  average the probability change.
- ``shapley``: sampled Shapley values over feature coalitions, using the
  background mean as the absent-feature reference.
- ``surrogate``: a local linear surrogate fitted on seeded perturbations
  of the row, weighted by distance.

All randomness flows from a caller-supplied ``random.Random`` so results
are deterministic.
"""

import math
import random


def _background_means(background):
    n = len(background)
    d = len(background[0])
    return [sum(r[j] for r in background) / n for j in range(d)]


def explain_loco(model, background, row):
    means = _background_means(background)
    base = model.predict_proba_one(row)
    out = []
    for j in range(len(row)):
        modified = list(row)
        modified[j] = means[j]
        out.append(base - model.predict_proba_one(modified))
    return out


def explain_permutation(model, background, row, rng, n_repeats=6):
    base = model.predict_proba_one(row)
    d = len(row)
    out = []
    for j in range(d):
        total = 0.0
        for _ in range(n_repeats):
            modified = list(row)
            donor = background[rng.randrange(len(background))]
            modified[j] = donor[j]
            total += base - model.predict_proba_one(modified)
        out.append(total / n_repeats)
    return out


def explain_shapley(model, background, row, rng, n_permutations=24):
    means = _background_means(background)
    d = len(row)

    def value(subset):
        modified = [
            row[j] if j in subset else means[j] for j in range(d)
        ]
        return model.predict_proba_one(modified)

    contrib = [0.0] * d
    for _ in range(n_permutations):
        order = list(range(d))
        rng.shuffle(order)
        subset = set()
        prev = value(subset)
        for j in order:
            subset.add(j)
            cur = value(subset)
            contrib[j] += cur - prev
            prev = cur
    return [c / n_permutations for c in contrib]


def _solve_linear(a, b):
    """Gaussian elimination for a small dense system a x = b."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            m[col][col] += 1e-8
            pivot = col
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for r in range(n):
            if r != col and abs(m[r][col]) > 0.0:
                factor = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def explain_surrogate(model, background, row, rng, n_samples=160, scale=0.7):
    d = len(row)
    samples = []
    targets = []
    weights = []
    for _ in range(n_samples):
        perturbed = [row[j] + rng.gauss(0.0, scale) for j in range(d)]
        samples.append(perturbed)
        targets.append(model.predict_proba_one(perturbed))
        dist2 = sum((perturbed[j] - row[j]) ** 2 for j in range(d))
        weights.append(math.exp(-dist2 / (2.0 * scale * scale)))
    # Weighted least squares with intercept: solve (X^T W X) beta = X^T W y.
    p = d + 1
    a = [[0.0] * p for _ in range(p)]
    bvec = [0.0] * p
    for s in range(n_samples):
        xs = [1.0] + samples[s]
        w = weights[s]
        for i in range(p):
            for j in range(p):
                a[i][j] += w * xs[i] * xs[j]
            bvec[i] += w * xs[i] * targets[s]
    for i in range(p):
        a[i][i] += 1e-6
    beta = _solve_linear(a, bvec)
    # Attribution: coefficient x (row value - background mean), so the
    # vector is on the same signed scale as the other explainers.
    means = _background_means(background)
    return [beta[j + 1] * (row[j] - means[j]) for j in range(d)]


EXPLAINERS = ("loco", "permutation", "shapley", "surrogate")


def explain_all(model, background, row, seed):
    """Run every explainer on one row; returns {name: attribution}."""
    rng = random.Random(seed)
    return {
        "loco": explain_loco(model, background, row),
        "permutation": explain_permutation(model, background, row, rng),
        "shapley": explain_shapley(model, background, row, rng),
        "surrogate": explain_surrogate(model, background, row, rng),
    }
