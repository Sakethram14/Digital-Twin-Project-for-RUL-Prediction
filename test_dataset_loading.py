from data.data_loader import load_all_cmapss
from data.window_generator import generate_temporal_windows
from data.window_to_state import window_to_state
from ml.inference import SimpleHealthModel

# Load all C-MAPSS subsets
df = load_all_cmapss("data/cmapss", split="train")

print("Total rows:", len(df))
print("Total engines:", df["engine_id"].nunique())

# Generate temporal windows
windows = generate_temporal_windows(df, window_size=20)
print("Total temporal samples:", len(windows))

# Prepare one window for inference
sensor_cols = [c for c in df.columns if "sensor" in c]
state_window = window_to_state(windows[0], sensor_cols)

# Run inference
model = SimpleHealthModel()
health = model.predict(state_window)

print("Predicted Health:", health)
print("Inference Latency:", model.last_latency)
