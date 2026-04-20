def generate_temporal_windows(df, window_size=20):
    windows = []

    sensor_cols = [c for c in df.columns if "sensor" in c]

    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
        values = engine_df[sensor_cols].values

        for i in range(len(values) - window_size + 1):
            windows.append(values[i:i + window_size])

    return windows
