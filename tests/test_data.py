import pytest

from disagreement_signal import data as data_mod


def test_generate_shape():
    rows, labels, regions = data_mod.generate(50, seed=1)
    assert len(rows) == 50
    assert all(len(r) == data_mod.N_FEATURES for r in rows)
    assert all(y in (0, 1) for y in labels)
    assert all(r in data_mod.REGIONS for r in regions)


def test_generate_deterministic():
    a = data_mod.generate(30, seed=3)
    b = data_mod.generate(30, seed=3)
    assert a == b


def test_generate_seed_changes_data():
    a = data_mod.generate(30, seed=3)
    b = data_mod.generate(30, seed=4)
    assert a[0] != b[0]


def test_split_sizes():
    train, val, test = data_mod.split_dataset(seed=7, n_train=10, n_val=5, n_test=6)
    assert len(train[0]) == 10
    assert len(val[0]) == 5
    assert len(test[0]) == 6


def test_split_deterministic():
    a = data_mod.split_dataset(seed=7, n_train=10, n_val=5, n_test=6)
    b = data_mod.split_dataset(seed=7, n_train=10, n_val=5, n_test=6)
    assert a == b


def test_regions_present_in_large_sample():
    _rows, _labels, regions = data_mod.generate(1500, seed=7)
    counts = data_mod.region_counts(regions)
    assert counts.get("interaction", 0) > 0
    assert counts.get("shift", 0) > 0
    assert counts.get("normal", 0) > 0


def test_latent_score_region_labels():
    # Both x0, x1 high -> interaction pocket.
    _score, region = data_mod.latent_score([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert region == "interaction"
    # Nothing planted -> normal.
    _score, region = data_mod.latent_score([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert region == "normal"


def test_sigmoid_bounds():
    assert 0.0 < data_mod.sigmoid(-20.0) < 0.01
    assert 0.99 < data_mod.sigmoid(20.0) < 1.0
    assert data_mod.sigmoid(0.0) == pytest.approx(0.5)
