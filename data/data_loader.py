import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

COLUMNS = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)
def normalize_sensors(df):
    sensor_cols = [c for c in df.columns if "sensor" in c]
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df
    
def load_cmapss_subset(base_path, subset, split="train"):
    file_name = f"{split}_{subset}.txt"
    file_path = os.path.join(base_path, subset, file_name)

    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = COLUMNS
    return df


def load_all_cmapss(base_path, split="train"):
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    dfs = []

    for subset in subsets:
        df = load_cmapss_subset(base_path, subset, split)
        df["subset"] = subset
        dfs.append(df)

    # Combine all subsets
    combined_df = pd.concat(dfs, ignore_index=True)

    # -------- NORMALIZATION (IMPORTANT EDIT) --------
    sensor_cols = [c for c in combined_df.columns if "sensor" in c]

    scaler = MinMaxScaler()
    combined_df[sensor_cols] = scaler.fit_transform(combined_df[sensor_cols])
    # -----------------------------------------------

    return combined_df
