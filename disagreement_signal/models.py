"""Simple deterministic stdlib classifiers.

Three model classes with different inductive biases, each exposing
``predict_proba_one(row) -> float`` (probability of class 1):

- Logistic regression trained by full-batch gradient descent.
- A small CART-style decision tree (Gini splits, depth-limited).
- k-nearest neighbours on standardised features.

No third-party code; everything is seeded and deterministic.
"""

import math


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticRegressionGD(object):
    def __init__(self, n_features, lr=0.1, epochs=300, l2=0.01):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.means = [0.0] * n_features
        self.stds = [1.0] * n_features

    def _standardise(self, row):
        return [
            (row[j] - self.means[j]) / self.stds[j] for j in range(self.n_features)
        ]

    def fit(self, rows, labels):
        n = len(rows)
        d = self.n_features
        self.means = [sum(r[j] for r in rows) / n for j in range(d)]
        self.stds = []
        for j in range(d):
            var = sum((r[j] - self.means[j]) ** 2 for r in rows) / n
            self.stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
        xs = [self._standardise(r) for r in rows]
        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for i in range(n):
                z = self.bias + sum(self.weights[j] * xs[i][j] for j in range(d))
                err = _sigmoid(z) - labels[i]
                for j in range(d):
                    grad_w[j] += err * xs[i][j]
                grad_b += err
            for j in range(d):
                self.weights[j] -= self.lr * (grad_w[j] / n + self.l2 * self.weights[j])
            self.bias -= self.lr * (grad_b / n)
        return self

    def predict_proba_one(self, row):
        x = self._standardise(row)
        z = self.bias + sum(self.weights[j] * x[j] for j in range(self.n_features))
        return _sigmoid(z)


class _Node(object):
    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.proba = 0.0


def _gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    p = sum(labels) / float(n)
    return 2.0 * p * (1.0 - p)


class DecisionTree(object):
    def __init__(self, max_depth=5, min_leaf=12):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root = None
        self.n_features = 0

    def fit(self, rows, labels):
        self.n_features = len(rows[0])
        idx = list(range(len(rows)))
        self.root = self._build(rows, labels, idx, 0)
        return self

    def _build(self, rows, labels, idx, depth):
        node = _Node()
        node.proba = sum(labels[i] for i in idx) / float(len(idx))
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return node
        base = _gini([labels[i] for i in idx])
        best = None
        for j in range(self.n_features):
            values = sorted(set(rows[i][j] for i in idx))
            if len(values) < 2:
                continue
            for t in range(len(values) - 1):
                thr = 0.5 * (values[t] + values[t + 1])
                left = [i for i in idx if rows[i][j] <= thr]
                right = [i for i in idx if rows[i][j] > thr]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                gain = base - (
                    len(left) * _gini([labels[i] for i in left])
                    + len(right) * _gini([labels[i] for i in right])
                ) / float(len(idx))
                if best is None or gain > best[0]:
                    best = (gain, j, thr, left, right)
        if best is None or best[0] <= 1e-9:
            return node
        _, j, thr, left, right = best
        node.feature = j
        node.threshold = thr
        node.left = self._build(rows, labels, left, depth + 1)
        node.right = self._build(rows, labels, right, depth + 1)
        return node

    def predict_proba_one(self, row):
        node = self.root
        while node.feature is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.proba


class KNearestNeighbours(object):
    def __init__(self, k=15):
        self.k = k
        self.rows = []
        self.labels = []
        self.means = []
        self.stds = []

    def fit(self, rows, labels):
        n = len(rows)
        d = len(rows[0])
        self.means = [sum(r[j] for r in rows) / n for j in range(d)]
        self.stds = []
        for j in range(d):
            var = sum((r[j] - self.means[j]) ** 2 for r in rows) / n
            self.stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
        self.rows = [self._z(r) for r in rows]
        self.labels = list(labels)
        return self

    def _z(self, row):
        return [(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))]

    def predict_proba_one(self, row):
        z = self._z(row)
        dists = []
        for i, train_row in enumerate(self.rows):
            d2 = sum((z[j] - train_row[j]) ** 2 for j in range(len(z)))
            dists.append((d2, self.labels[i]))
        dists.sort(key=lambda t: t[0])
        neighbours = dists[: self.k]
        return sum(lbl for _, lbl in neighbours) / float(len(neighbours))


def train_all(train_rows, train_labels):
    """Train the three model classes; returns {name: model}.

    k-NN is fitted on a 300-row prefix of the training set: every
    explainer call re-runs prediction, and a full 800-row neighbourhood
    scan per call makes the evaluation needlessly slow. The prefix is
    deterministic because the dataset itself is seeded.
    """
    d = len(train_rows[0])
    models = {
        "logistic": LogisticRegressionGD(d),
        "tree": DecisionTree(),
        "knn": KNearestNeighbours(),
    }
    models["logistic"].fit(train_rows, train_labels)
    models["tree"].fit(train_rows, train_labels)
    models["knn"].fit(train_rows[:300], train_labels[:300])
    return models


def predict_label(model, row):
    return 1 if model.predict_proba_one(row) >= 0.5 else 0


def accuracy(model, rows, labels):
    correct = 0
    for row, y in zip(rows, labels):
        if predict_label(model, row) == y:
            correct += 1
    return correct / float(len(rows))
