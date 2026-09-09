def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "average_response_time" in data
    assert "endpoint_requests" in data

    assert isinstance(data["total_requests"], int)
    assert isinstance(data["successful_requests"], int)
    assert isinstance(data["failed_requests"], int)
    assert isinstance(data["average_response_time"], float)
    assert isinstance(data["endpoint_requests"], dict)