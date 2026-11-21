import pytest

from disagreement_signal import data as data_mod
from disagreement_signal import models as models_mod


@pytest.fixture(scope="module")
def trained():
    train, _val, test = data_mod.split_dataset(
        seed=7, n_train=300, n_val=20, n_test=60
    )
    models = models_mod.train_all(train[0], train[1])
    return models, test


def test_train_all_returns_three_models(trained):
    models, _test = trained
    assert set(models.keys()) == {"logistic", "tree", "knn"}


def test_proba_in_unit_interval(trained):
    models, test = trained
    for model in models.values():
        for row in test[0][:10]:
            p = model.predict_proba_one(row)
            assert 0.0 <= p <= 1.0


def test_models_beat_chance(trained):
    models, test = trained
    for model in models.values():
        acc = models_mod.accuracy(model, test[0], test[1])
        assert acc > 0.55


def test_logistic_deterministic():
    rows, labels, _ = data_mod.generate(80, seed=5)
    a = models_mod.LogisticRegressionGD(8, epochs=50).fit(rows, labels)
    b = models_mod.LogisticRegressionGD(8, epochs=50).fit(rows, labels)
    assert a.weights == b.weights
    assert a.bias == b.bias


def test_tree_predicts_training_signal():
    rows, labels, _ = data_mod.generate(200, seed=5)
    tree = models_mod.DecisionTree().fit(rows, labels)
    acc = models_mod.accuracy(tree, rows, labels)
    assert acc > 0.6


def test_knn_k1_memorises():
    rows, labels, _ = data_mod.generate(40, seed=5)
    knn = models_mod.KNearestNeighbours(k=1).fit(rows, labels)
    acc = models_mod.accuracy(knn, rows, labels)
    assert acc == pytest.approx(1.0)


def test_predict_label_threshold():
    class Fixed(object):
        def predict_proba_one(self, row):
            return 0.5

    assert models_mod.predict_label(Fixed(), [0.0]) == 1
