import logging
import time

class Monitoring:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0

    def log_request(self, request):
        self.request_count += 1
        self.logger.info(f"Request: {request.method} {request.url}")
        request.state.start_time = time.time()

    def log_response_time(self, request):
        elapsed = time.time() - getattr(request.state, 'start_time', time.time())
        self.total_response_time += elapsed
        self.logger.info(f"Response time: {elapsed:.4f}s")

    def log_error(self, error):
        self.error_count += 1
        self.logger.error(f"Error: {error}")

    def get_metrics(self):
        avg_response_time = self.total_response_time / self.request_count if self.request_count else 0.0
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time": avg_response_time
        } 