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