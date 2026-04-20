# edge/edge_processor.py
"""
Edge Processing Module
- Selects important sensors
- Normalizes sensor values
- Outputs clean features for the Digital Twin
"""

# 1. Sensors selected at the edge (reduced feature set)
IMPORTANT_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_15", "sensor_17"
]

# 2. Approximate min–max ranges for normalization (CMAPSS FD001)
SENSOR_RANGES = {
    "sensor_2": (640, 645),
    "sensor_3": (1500, 1650),
    "sensor_4": (1350, 1450),
    "sensor_7": (550, 560),
    "sensor_8": (2385, 2390),
    "sensor_9": (9000, 9100),
    "sensor_11": (45, 50),
    "sensor_12": (520, 525),
    "sensor_15": (8.3, 8.6),
    "sensor_17": (390, 395)
}

# 3. Min–Max normalization function
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


# 4. Main edge processing function
def edge_process(payload):
    """
    Input  : Raw sensor payload from streamer
    Output : Normalized feature vector
    """

    processed_payload = {
        "engine_id": payload["engine_id"],
        "cycle": payload["cycle"],
        "features": {}
    }

    for sensor in IMPORTANT_SENSORS:
        raw_value = payload["sensors"][sensor]
        min_val, max_val = SENSOR_RANGES[sensor]
        processed_payload["features"][sensor] = normalize(
            raw_value, min_val, max_val
        )

    return processed_payload


# 5. Local test (ONLY for Day 2 testing)
if __name__ == "__main__":
    sample_payload = {
        "engine_id": 1,
        "cycle": 1,
        "sensors": {
            "sensor_2": 641.82,
            "sensor_3": 1589.7,
            "sensor_4": 1400.6,
            "sensor_7": 554.36,
            "sensor_8": 2388.06,
            "sensor_9": 9046.19,
            "sensor_11": 47.47,
            "sensor_12": 521.66,
            "sensor_15": 8.4195,
            "sensor_17": 392.0
        }
    }

    output = edge_process(sample_payload)
    print(output)
