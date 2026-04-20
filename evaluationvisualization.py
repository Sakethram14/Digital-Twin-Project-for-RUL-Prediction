import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_visualizations(csv_path="results/health_rul_log.csv", threshold=0.4):

    if not os.path.exists(csv_path):
        print("No CSV results file found.")
        return

    df = pd.read_csv(csv_path)

    os.makedirs("results/plots", exist_ok=True)

    # -------------------------------------------------
    # 1. Health Trend
    # -------------------------------------------------
    plt.figure()
    plt.plot(df["Step"], df["Health"], marker="o")
    plt.title("Health Score Trend")
    plt.xlabel("Inference Step")
    plt.ylabel("Health Score")
    plt.grid(True)
    plt.savefig("results/plots/health_trend.png")
    plt.close()

    # -------------------------------------------------
    # 2. RUL Trend
    # -------------------------------------------------
    plt.figure()
    plt.plot(df["Step"], df["RUL"], marker="o", color="orange")
    plt.title("Remaining Useful Life (RUL)")
    plt.xlabel("Inference Step")
    plt.ylabel("RUL")
    plt.grid(True)
    plt.savefig("results/plots/rul_trend.png")
    plt.close()

    # -------------------------------------------------
    # 3. Latency Trend
    # -------------------------------------------------
    plt.figure()
    plt.plot(df["Step"], df["Latency"], marker="o", color="green")
    plt.title("Inference Latency per Step")
    plt.xlabel("Inference Step")
    plt.ylabel("Latency (seconds)")
    plt.grid(True)
    plt.savefig("results/plots/latency_trend.png")
    plt.close()

    # -------------------------------------------------
    # 4. Health Distribution
    # -------------------------------------------------
    plt.figure()
    plt.hist(df["Health"], bins=15, color="purple")
    plt.title("Health Score Distribution")
    plt.xlabel("Health Score")
    plt.ylabel("Frequency")
    plt.savefig("results/plots/health_distribution.png")
    plt.close()

    # -------------------------------------------------
    # 5. Degradation Curve
    # -------------------------------------------------
    degradation = 1 - df["Health"]

    plt.figure()
    plt.plot(df["Step"], degradation, color="red")
    plt.title("System Degradation Over Time")
    plt.xlabel("Inference Step")
    plt.ylabel("Degradation Level")
    plt.grid(True)
    plt.savefig("results/plots/degradation_curve.png")
    plt.close()

    # -------------------------------------------------
    # 6. Decision Threshold Visualization
    # -------------------------------------------------
    plt.figure()
    plt.plot(df["Step"], df["Health"], label="Health")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Maintenance Threshold")
    plt.title("Health vs Maintenance Threshold")
    plt.xlabel("Inference Step")
    plt.ylabel("Health Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/decision_threshold.png")
    plt.close()

    print("✔ All visualizations saved in results/plots/")