from ml.inference import SimpleHealthModel
from evaluation.prediction_metrics import rmse, mae
from evaluation.system_metrics import temporal_consistency
from evaluation.logger import log_metrics

# -----------------------------
# Fake temporal state window
# -----------------------------
state_window = [
    {"s1": 0.2, "s2": 0.3, "s3": 0.25},
    {"s1": 0.22, "s2": 0.32, "s3": 0.27},
    {"s1": 0.25, "s2": 0.35, "s3": 0.30},
]

# Ground truth (dummy, for testing)
y_true = [0.75]
y_pred = []

# -----------------------------
# Run model
# -----------------------------
model = SimpleHealthModel()
prediction = model.predict(state_window)
y_pred.append(prediction)

# -----------------------------
# Evaluate
# -----------------------------
metrics = {
    "Predicted Health": prediction,
    "RMSE": rmse(y_true, y_pred),
    "MAE": mae(y_true, y_pred),
    "Temporal Consistency": temporal_consistency(y_pred),
    "Inference Latency (s)": model.last_latency
}

log_metrics(metrics, title="Inference + Evaluation Test")
