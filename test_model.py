"""Automated tests for the Iris machine-learning model."""
from sklearn.datasets import load_iris
from model import train_model


def test_model_trains_and_has_reasonable_accuracy():
    model, accuracy = train_model()
    assert model is not None
    assert 0.0 <= accuracy <= 1.0
    assert accuracy >= 0.80


def test_model_predicts_valid_iris_classes():
    iris = load_iris()
    model, _ = train_model()
    predictions = model.predict(iris.data[:5])
    assert len(predictions) == 5
    assert set(predictions).issubset(set(iris.target))
