import time
import numpy as np

def measure_latency(model, input_window):
    """
    Measures end-to-end inference latency (seconds)
    """
    start_time = time.time()
    _ = model.predict(input_window)
    end_time = time.time()
    return end_time - start_time

def temporal_consistency(predictions):
    """
    Measures smoothness of predictions over time.
    Lower value = smoother behavior.
    """
    predictions = np.array(predictions)
    if len(predictions) < 2:
        return 0.0
    return np.mean(np.abs(np.diff(predictions)))

def state_sync_error(edge_state, twin_state):
    """
    Measures difference between edge features and twin state
    """
    edge_state = np.array(edge_state)
    twin_state = np.array(twin_state)
    return np.mean(np.abs(edge_state - twin_state))

def closed_loop_check(twin_state_dict):
    """
    Verifies if prediction is fed back to the Digital Twin
    """
    return (
        "latest_prediction" in twin_state_dict and
        twin_state_dict["latest_prediction"] is not None
    )
