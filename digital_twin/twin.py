# digital_twin/twin.py

from collections import deque

class DigitalTwin:
    """
    Digital Twin Model for a Physical Engine
    """

    def __init__(self, engine_id, window_size=30):
        self.engine_id = engine_id
        self.window_size = window_size

        # Sliding window of recent feature vectors
        self.state_window = deque(maxlen=window_size)

        # Latest system state
        self.current_state = None

        # Latest prediction (to be filled by ML model)
        self.latest_prediction = None

    def update_state(self, edge_payload):
        """
        Update the Digital Twin with new edge-processed data
        """
        self.current_state = edge_payload
        self.state_window.append(edge_payload["features"])

    def get_state_window(self):
        """
        Return time-windowed feature data (for ML inference)
        """
        return list(self.state_window)

    def update_prediction(self, prediction):
        """
        Update twin with ML prediction (RUL / health score)
        """
        self.latest_prediction = prediction

    def get_twin_status(self):
        """
        Return full Digital Twin status (for dashboard)
        """
        return {
            "engine_id": self.engine_id,
            "current_cycle": self.current_state["cycle"]
            if self.current_state else None,
            "latest_prediction": self.latest_prediction,
            "window_length": len(self.state_window)
        }
if __name__ == "__main__":
    # Create a Digital Twin instance
    twin = DigitalTwin(engine_id=1, window_size=5)

    # Simulate edge-processed input
    for cycle in range(1, 8):
        sample_edge_payload = {
            "engine_id": 1,
            "cycle": cycle,
            "features": {
                "sensor_2": 0.3 + cycle * 0.01,
                "sensor_3": 0.5 + cycle * 0.01
            }
        }
        twin.update_state(sample_edge_payload)

    print("Twin status:", twin.get_twin_status())
    print("State window:", twin.get_state_window())
