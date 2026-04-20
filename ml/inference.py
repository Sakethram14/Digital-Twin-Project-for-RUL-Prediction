# ml/inference.py
import numpy as np
import time

class SimpleHealthModel:
    """
    Simple ML model for health estimation (MVP purpose)
    Output is designed to support evaluation metrics
    """

    def __init__(self):

        # -------------------------------------------------
        # Model / Learning Parameters
        # -------------------------------------------------
        self.model_name = "SimpleHealthModel_MVP"
        self.inference_type = "Window-Based Temporal Inference"
        self.degradation_method = "Mean Sensor Degradation"
        self.health_scale_min = 0.0
        self.health_scale_max = 1.0

        # System metric tracking
        self.last_latency = None


    def print_model_config(self):
        """Display model configuration"""
        print("\nModel Configuration")
        print(f"   ✔ Model Name          : {self.model_name}")
        print(f"   ✔ Inference Type      : {self.inference_type}")
        print(f"   ✔ Degradation Method  : {self.degradation_method}")
        print(f"   ✔ Health Range        : {self.health_scale_min} - {self.health_scale_max}")


    def predict(self, state_window):
        """
        Input  : Sliding window of feature dictionaries from Digital Twin
        Output : Health score (float, 0–1)
        """

        if state_window is None or len(state_window) == 0:
            return None

        start_time = time.time()

        # Convert list of feature dicts → numeric array
        try:
            data = np.array(
                [list(features.values()) for features in state_window],
                dtype=np.float32
            )
        except Exception as e:
            raise ValueError(f"Invalid state window format: {e}")

        # -------------------------------------------------
        # Simple heuristic (MVP logic)
        # -------------------------------------------------

        # Mean degradation across window
        mean_value = np.mean(data)

        # Health estimation (inverse degradation)
        health_score = 1.0 - mean_value

        # Clamp health between valid range
        health_score = np.clip(
            health_score,
            self.health_scale_min,
            self.health_scale_max
        )

        # Store inference latency
        self.last_latency = time.time() - start_time

        return float(round(health_score, 3))