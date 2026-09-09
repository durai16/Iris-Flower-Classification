def test_predict_batch_valid(client):
    payload = {
        "inputs": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.2,
                "sepal_width": 2.8,
                "petal_length": 4.8,
                "petal_width": 1.8
            }
        ]
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 200


def test_predict_batch_oversized(client):
    payload = {
        "inputs": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        ] * 101
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 500


def test_missing_api_key(client):
    client.headers.pop("X-API-Key", None)

    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 401


def test_invalid_api_key(client):
    client.headers["X-API-Key"] = "wrong-api-key"

    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 401


def test_unexpected_extra_field(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "unexpected_field": "test"
        }
    )

    assert response.status_code == 422


def test_negative_value_rejected(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 422