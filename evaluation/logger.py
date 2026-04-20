def log_metrics(metrics: dict, title="Evaluation Metrics"):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("=" * 40 + "\n")
