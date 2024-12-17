"""
tests pytest
"""

import numpy as np
from LinearRegressions import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    LogisticRegression,
)


def test_LinearRegression():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.allclose(predictions, y)


def test_LinearRegression_high_dim():
    X = np.array([[1, 2, 3, 4, 5], [7, 3, 2, 1, 8]]).T
    y = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.allclose(predictions, y)


def test_RidgeRegression():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    model = RidgeRegression(alpha=0.1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=0.2)


def test_LassoRegression():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[1], [2], [3], [4], [5]])
    model = LassoRegression(alpha=0.1, learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=0.2)


def test_ElasticNet():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[1], [2], [3], [4], [5]])
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=0.1)


def test_LogisticRegression():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[0], [0], [1], [1], [1]])
    model = LogisticRegression(alpha=0, learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y)
