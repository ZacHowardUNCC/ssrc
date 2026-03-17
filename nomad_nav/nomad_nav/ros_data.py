import time


class ROSData:
    """Container for ROS message data with a staleness timeout.

    Uses wall-clock time (time.monotonic) instead of rospy.get_time()
    so it works identically under rclpy without requiring a node handle.
    """

    def __init__(self, timeout: float = 3.0, queue_size: int = 1, name: str = ""):
        self.timeout = timeout
        self.last_time_received = float("-inf")
        self.queue_size = queue_size
        self.data = None
        self.name = name

    def get(self):
        return self.data

    def set(self, data):
        time_waited = time.monotonic() - self.last_time_received
        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timeout:
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = time.monotonic()

    def is_valid(self, verbose: bool = False) -> bool:
        time_waited = time.monotonic() - self.last_time_received
        valid = time_waited < self.timeout
        if self.queue_size > 1:
            valid = valid and self.data is not None and len(self.data) == self.queue_size
        if verbose and not valid:
            print(
                f"Not receiving {self.name} data for {time_waited:.1f}s "
                f"(timeout: {self.timeout}s)"
            )
        return valid
