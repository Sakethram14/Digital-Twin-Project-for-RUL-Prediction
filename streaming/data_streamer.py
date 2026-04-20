# streaming/data_streamer.py

import pandas as pd
import time

# Project imports
from utils.config import DATA_PATH, STREAM_DELAY, COLUMNS
from edge.edge_processor import edge_process
from digital_twin.twin import DigitalTwin
from ml.inference import SimpleHealthModel


# --------------------------------------------------
# Load CMAPSS dataset
# --------------------------------------------------
def load_cmapss_data():
    """
    Load CMAPSS dataset and assign column names
    """
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None
    )
    df.columns = COLUMNS
    return df


# --------------------------------------------------
# Stream data through Edge → Digital Twin → ML
# --------------------------------------------------
def stream_data(df, twin, health_model):
    print("🚀 Streaming → Edge → Digital Twin → ML Inference...\n")

    for _, row in df.iterrows():

        # ---------------- Physical System ----------------
        payload = {
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
            "sensors": row[5:].to_dict()
        }

        # ---------------- Edge Processing ----------------
        edge_output = edge_process(payload)

        # ---------------- Digital Twin Update ----------------
        twin.update_state(edge_output)

        # ---------------- ML Inference (Cloud) ----------------
        state_window = twin.get_state_window()
        prediction = health_model.predict(state_window)

        # Update twin with prediction
        twin.update_prediction(prediction)

        # ---------------- Output (for demo / logging) ----------------
        print("TWIN STATUS:", twin.get_twin_status())

        time.sleep(STREAM_DELAY)


# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":

    # Load data
    df = load_cmapss_data()

    # Initialize Digital Twin
    twin = DigitalTwin(engine_id=1, window_size=30)

    # Initialize ML model
    health_model = SimpleHealthModel()

    # Start streaming pipeline
    stream_data(df, twin, health_model)
