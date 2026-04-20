# utils/config.py

STREAM_DELAY = 0.5
DATA_PATH = "data/CMAPSSData/train_FD001.txt"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]
