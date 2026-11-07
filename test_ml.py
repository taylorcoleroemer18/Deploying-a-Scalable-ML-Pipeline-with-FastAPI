import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


def test_train_model():
    """Check that training returns a Random Forest classifier."""
    X = np.array([
        [25, 40],
        [35, 50],
        [45, 60],
        [55, 70],
        [30, 45],
        [50, 65]
    ])
    y = np.array([0, 0, 1, 1, 0, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_inference():
    """Check that inference returns the expected predictions."""
    X = np.array([
        [25, 40],
        [35, 50],
        [45, 60],
        [55, 70],
        [30, 45],
        [50, 65]
    ])
    y = np.array([0, 0, 1, 1, 0, 1])

    model = train_model(X, y)
    predictions = inference(model, X)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_compute_model_metrics():
    """Check that precision, recall, and F1 are calculated correctly."""
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert np.isclose(precision, 2 / 3)
    assert np.isclose(recall, 2 / 3)
    assert np.isclose(f1, 2 / 3)
