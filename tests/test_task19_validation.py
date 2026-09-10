import pytest
from task19_validation import predict_new_iris


def test_valid_iris_input():
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    result = predict_new_iris(data)

    assert result["prediction"] == "Iris Setosa"
    assert 0 <= result["confidence"] <= 1


def test_missing_field():
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }

    with pytest.raises(ValueError, match="Missing required fields"):
        predict_new_iris(data)


def test_negative_value():
    data = {
        "sepal_length": 5.1,
        "sepal_width": -3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with pytest.raises(ValueError, match="must be greater than 0"):
        predict_new_iris(data)


def test_wrong_data_type():
    data = {
        "sepal_length": "five",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with pytest.raises(ValueError, match="must be a numeric value"):
        predict_new_iris(data)