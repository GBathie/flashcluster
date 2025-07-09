from flashcluster import compute_clustering
import numpy as np
import pytest


def test_f32():
    data = np.random.random((100, 20)).astype(dtype=np.float32)
    _um = compute_clustering(data, 1.5, "fast")


def test_f64():
    data = np.random.random((100, 20)).astype(dtype=np.float64)
    _um = compute_clustering(data, 1.5, "fast")


def test_1d_fails():
    with pytest.raises(ValueError):
        data = np.random.random(100).astype(dtype=np.float16)
        _um = compute_clustering(data, 1.5, "fast")


def test_3d_fails():
    with pytest.raises(ValueError):
        data = np.random.random((10, 10, 10)).astype(dtype=np.float16)
        _um = compute_clustering(data, 1.5, "fast")


def test_c_less_1_fails():
    with pytest.raises(ValueError):
        data = np.random.random((10, 10)).astype(dtype=np.float16)
        _um = compute_clustering(data, 0.9, "fast")
