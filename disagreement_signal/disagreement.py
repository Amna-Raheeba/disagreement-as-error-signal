"""The disagreement vector and the composite distrust score.

Given the attribution vectors of several explainers for the same
prediction, disagreement is measured three ways:

- ``rank_spread``: (1 - mean pairwise Spearman rank correlation) / 2,
  normalised to [0, 1] (do they rank features in the same order?).
- ``topk_gap``: 1 - mean pairwise overlap of the top-k features by
  absolute attribution (do they name the same important features?).
- ``sign_conflict``: fraction of features whose attribution sign differs
  across explainers, weighted by the largest absolute attribution for
  that feature (do they push in opposite directions on features that
  matter?).

The composite ``disagreement_score`` is the unweighted mean of the three
components, in [0, 1]. It needs no labels and is available at inference
time -- that is the whole point.
"""


def ranks(values):
    """Average ranks (0-based) of a list of values."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j)
        for t in range(i, j + 1):
            out[order[t]] = avg
        i = j + 1
    return out


def spearman(a, b):
    ra = ranks(a)
    rb = ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n))
    db = sum((rb[i] - mb) ** 2 for i in range(n))
    if da <= 0.0 or db <= 0.0:
        return 1.0
    return num / ((da * db) ** 0.5)


def _pairs(items):
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def topk_set(attribution, k):
    order = sorted(
        range(len(attribution)), key=lambda i: -abs(attribution[i])
    )
    return set(order[:k])


def disagreement_vector(attributions, topk=3):
    """attributions: {explainer_name: [float]} -> disagreement dict."""
    names = sorted(attributions.keys())
    vecs = [attributions[name] for name in names]
    d = len(vecs[0])

    corrs = [spearman(a, b) for a, b in _pairs(vecs)]
    rank_spread = (
        (1.0 - (sum(corrs) / len(corrs))) / 2.0 if corrs else 0.0
    )

    overlaps = []
    for a, b in _pairs(vecs):
        sa = topk_set(a, topk)
        sb = topk_set(b, topk)
        overlaps.append(len(sa & sb) / float(topk))
    topk_gap = 1.0 - (sum(overlaps) / len(overlaps)) if overlaps else 0.0

    conflict_weight = 0.0
    total_weight = 0.0
    for j in range(d):
        signs = set()
        for v in vecs:
            if v[j] > 1e-9:
                signs.add(1)
            elif v[j] < -1e-9:
                signs.add(-1)
        weight = max(abs(v[j]) for v in vecs)
        total_weight += weight
        if len(signs) > 1:
            conflict_weight += weight
    sign_conflict = (
        conflict_weight / total_weight if total_weight > 0 else 0.0
    )

    score = (rank_spread + topk_gap + sign_conflict) / 3.0
    return {
        "rank_spread": rank_spread,
        "topk_gap": topk_gap,
        "sign_conflict": sign_conflict,
        "disagreement_score": score,
    }
