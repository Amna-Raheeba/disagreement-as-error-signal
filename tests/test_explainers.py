import random

import pytest

from disagreement_signal import data as data_mod
from disagreement_signal import models as models_mod
from disagreement_signal.explainers import (
    EXPLAINERS,
    explain_all,
    explain_loco,
    explain_permutation,
    explain_shapley,
    explain_surrogate,
)


@pytest.fixture(scope="module")
def world():
    train, _val, test = data_mod.split_dataset(
        seed=7, n_train=200, n_val=10, n_test=10
    )
    model = models_mod.LogisticRegressionGD(8).fit(train[0], train[1])
    return model, train[0], test[0][0]


def test_loco_length(world):
    model, background, row = world
    out = explain_loco(model, background, row)
    assert len(out) == 8


def test_permutation_length_and_determinism(world):
    model, background, row = world
    a = explain_permutation(model, background, row, random.Random(1))
    b = explain_permutation(model, background, row, random.Random(1))
    assert len(a) == 8
    assert a == b


def test_shapley_length_and_determinism(world):
    model, background, row = world
    a = explain_shapley(model, background, row, random.Random(1))
    b = explain_shapley(model, background, row, random.Random(1))
    assert len(a) == 8
    assert a == b


def test_shapley_efficiency_approx(world):
    # Sampled Shapley attributions should approximately sum to the gap
    # between the row prediction and the all-mean reference prediction.
    model, background, row = world
    means = [
        sum(r[j] for r in background) / len(background) for j in range(8)
    ]
    gap = model.predict_proba_one(row) - model.predict_proba_one(means)
    out = explain_shapley(model, background, row, random.Random(2), n_permutations=48)
    assert sum(out) == pytest.approx(gap, abs=0.05)


def test_surrogate_length_and_determinism(world):
    model, background, row = world
    a = explain_surrogate(model, background, row, random.Random(1))
    b = explain_surrogate(model, background, row, random.Random(1))
    assert len(a) == 8
    assert a == b


def test_explain_all_keys(world):
    model, background, row = world
    out = explain_all(model, background, row, 7)
    assert set(out.keys()) == set(EXPLAINERS)
    for vec in out.values():
        assert len(vec) == 8


def test_loco_zero_for_constant_model():
    class Constant(object):
        def predict_proba_one(self, row):
            return 0.3

    rows, _l, _r = data_mod.generate(10, seed=1)
    out = explain_loco(Constant(), rows, rows[0])
    assert all(v == 0.0 for v in out)
