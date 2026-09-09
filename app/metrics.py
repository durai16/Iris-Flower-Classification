from collections import defaultdict
from threading import Lock


class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.endpoint_requests = defaultdict(int)
        self._lock = Lock()

    def record_request(
        self,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        with self._lock:
            self.total_requests += 1
            self.total_response_time += duration

            self.endpoint_requests[endpoint] += 1

            if 200 <= status_code < 400:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    def get_metrics(self):
        with self._lock:
            average_response_time = (
                self.total_response_time / self.total_requests
                if self.total_requests > 0
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "average_response_time": round(
                    average_response_time,
                    4
                ),
                "endpoint_requests": dict(
                    self.endpoint_requests
                ),
            }


metrics = Metrics()