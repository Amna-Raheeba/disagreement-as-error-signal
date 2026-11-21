"""Evaluation metrics: AUC, threshold calibration, triage statistics.

AUC is the Mann-Whitney rank statistic (probability that a random
positive scores above a random negative), computed exactly with
average ranks for ties.
"""


def auc(scores, positives):
    """positives: list of bool (True = the event, here: model error)."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for t in range(i, j + 1):
            rank[order[t]] = avg
        i = j + 1
    n_pos = sum(1 for p in positives if p)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = sum(rank[i] for i in range(n) if positives[i])
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calibrate_threshold(scores, errors, target_review_rate=0.20):
    """Pick the distrust threshold on calibration data.

    The threshold is the score quantile that flags approximately
    ``target_review_rate`` of calibration cases for review. Returns the
    threshold value. Deterministic.
    """
    if not scores:
        return 0.0
    ordered = sorted(scores)
    idx = int(round((1.0 - target_review_rate) * (len(ordered) - 1)))
    return ordered[idx]


def triage_stats(scores, errors, threshold):
    """Apply a threshold; report what the review queue looks like."""
    n = len(scores)
    flagged = [i for i in range(n) if scores[i] >= threshold]
    n_errors = sum(1 for e in errors if e)
    flagged_errors = sum(1 for i in flagged if errors[i])
    return {
        "n": n,
        "review_rate": len(flagged) / float(n) if n else 0.0,
        "n_flagged": len(flagged),
        "error_capture": (
            flagged_errors / float(n_errors) if n_errors else 0.0
        ),
        "flagged_error_rate": (
            flagged_errors / float(len(flagged)) if flagged else 0.0
        ),
        "unflagged_error_rate": (
            (n_errors - flagged_errors) / float(n - len(flagged))
            if n - len(flagged) > 0
            else 0.0
        ),
        "overall_error_rate": n_errors / float(n) if n else 0.0,
    }


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def rank_normalise(values):
    """Map values to percentile ranks in [0, 1] (average ranks for ties)."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) / float(n - 1)
        for t in range(i, j + 1):
            out[order[t]] = avg
        i = j + 1
    return out
