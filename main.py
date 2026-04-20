import sys
import os
import time
import csv
from datetime import datetime
import yaml
import argparse

# -------------------------------------------------
# Ensure project root is on Python path
# -------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------
# Imports
# -------------------------------------------------
from data.data_loader import load_all_cmapss
from data.window_generator import generate_temporal_windows
from data.window_to_state import window_to_state

from ml.inference import SimpleHealthModel

from evaluation.prediction_metrics import rmse, mae
from evaluation.system_metrics import temporal_consistency
from evaluation.logger import log_metrics
from evaluationvisualization import generate_visualizations

# -------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATASET_PATH     = config["dataset_path"]
WINDOW_SIZE      = config["window_size"]
HEALTH_THRESHOLD = config["health_threshold"]
MAX_RUL          = config["max_rul"]
STREAM_DELAY     = config["stream_delay_sec"]

# -------------------------------------------------
# CLI ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Digital Twin MVP Runner")
parser.add_argument("--mode", default="realtime", choices=["realtime", "batch"])
parser.add_argument("--steps", type=int, default=50)
args = parser.parse_args()

MODE = args.mode
MAX_STEPS = args.steps

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    print("\n============================================")
    print(" Edge–Cloud Digital Twin MVP (Final Version)")
    print(" End-to-End System Validation")
    print("============================================")
    print(f"Execution Mode : {MODE.upper()}")
    print(f"Steps Executed : {MAX_STEPS}\n")

    # -------------------------------------------------
    # 1. LOAD DATASETS
    # -------------------------------------------------
    print("1️⃣ Loading C-MAPSS Datasets...")

    train_df = load_all_cmapss(DATASET_PATH, split="train")
    test_df  = load_all_cmapss(DATASET_PATH, split="test")

    print(f"   ✔ Train rows        : {len(train_df)}")
    print(f"   ✔ Test rows         : {len(test_df)}")
    print(f"   ✔ Engines (train)   : {train_df['engine_id'].nunique()}")
    print(f"   ✔ Engines (test)    : {test_df['engine_id'].nunique()}")

    # -------------------------------------------------
    # 2. TEMPORAL WINDOW GENERATION
    # -------------------------------------------------
    print("\n2️⃣ Generating Temporal Windows...")

    train_windows = generate_temporal_windows(train_df, WINDOW_SIZE)
    test_windows  = generate_temporal_windows(test_df, WINDOW_SIZE)

    print(f"   ✔ Train samples     : {len(train_windows)}")
    print(f"   ✔ Test samples      : {len(test_windows)}")

    # -------------------------------------------------
    # 3. DIGITAL TWIN INFERENCE
    # -------------------------------------------------
    print("\n3️⃣ Running Digital Twin Inference...")

    model = SimpleHealthModel()
    model.print_model_config()
    sensor_cols = [c for c in train_df.columns if "sensor" in c]

    health_trend  = []
    latency_trend = []
    rul_trend     = []

    for i in range(min(MAX_STEPS, len(test_windows))):
        state_window = window_to_state(test_windows[i], sensor_cols)
        health = model.predict(state_window)

        health_trend.append(health)
        latency_trend.append(model.last_latency)

        rul = int(health * MAX_RUL)
        rul_trend.append(rul)

        if MODE == "realtime":
            print(f"[{datetime.now()}] Health: {health:.3f}")
            time.sleep(STREAM_DELAY)

    print(f"   ✔ Health trend length : {len(health_trend)}")

    # -------------------------------------------------
    # 4. CLOSED-LOOP DECISION LOGIC
    # -------------------------------------------------
    print("\n4️⃣ Closed-Loop Decision Logic...")

    final_health = health_trend[-1]
    decision = (
        "⚠ Maintenance Required"
        if final_health < HEALTH_THRESHOLD
        else "✅ System Healthy"
    )

    print(f"   ✔ Final Health : {final_health:.3f}")
    print(f"   ✔ Decision     : {decision}")

    # -------------------------------------------------
    # 5. EVALUATION METRICS
    # -------------------------------------------------
    print("\n5️⃣ Evaluation Metrics...")

    y_true = [0.5] * len(health_trend)  # MVP reference

    metrics = {
        "RMSE": rmse(y_true, health_trend),
        "MAE": mae(y_true, health_trend),
        "Temporal Consistency": temporal_consistency(health_trend),
        "Avg Inference Latency (s)": sum(latency_trend) / len(latency_trend),
        "Final Health Score": final_health
    }

    log_metrics(metrics, title="Final MVP Evaluation")

    # -------------------------------------------------
    # 5b. BASELINE COMPARISON (NO TEMPORAL CONTEXT)
    # -------------------------------------------------
    baseline_health = []

    for i in range(min(MAX_STEPS, len(test_df))):
        snapshot = test_df.iloc[i:i+1][sensor_cols]
        state = [snapshot.iloc[0].to_dict()]
        baseline_health.append(model.predict(state))

    baseline_rmse = rmse(y_true[:len(baseline_health)], baseline_health)

    print("\nBaseline Comparison")
    print(f"   ✔ RMSE (Temporal) : {metrics['RMSE']:.4f}")
    print(f"   ✔ RMSE (Snapshot) : {baseline_rmse:.4f}")

    # -------------------------------------------------
    # 6. SAVE RESULTS TO CSV
    # -------------------------------------------------
    print("\n6️⃣ Saving Results to CSV...")

    os.makedirs("results", exist_ok=True)
    csv_path = "results/health_rul_log.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Health", "RUL", "Latency"])
        for i in range(len(health_trend)):
            writer.writerow([i, health_trend[i], rul_trend[i], latency_trend[i]])

    print(f"   ✔ Results saved to {csv_path}")
    # -------------------------------------------------
# 7. GENERATE VISUALIZATIONS
# -------------------------------------------------
    print("\n7️⃣ Generating Visualizations...")
    generate_visualizations("results/health_rul_log.csv", HEALTH_THRESHOLD)

    # -------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------
    print("\n============================================")
    print("✅ MVP END-TO-END EXECUTION SUCCESSFUL")
    print("✅ Dataset → DT → ML → Evaluation → Decision")
    print("============================================\n")


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
