def test_v1_and_v2_have_different_response_shapes(client):
    
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # V1 contract
    assert "prediction" in v1_data
    assert "confidence" in v1_data
    assert "request_id" in v1_data

    # V2 contract
    assert "prediction" in v2_data
    assert "probabilities" in v2_data
    assert "model_version" in v2_data
    assert "request_id" in v2_data

    # Breaking change:
    assert "confidence" not in v2_data
    assert "probabilities" not in v1_data