import pandas as pd


CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def preprocess_cmapss(df):
    df = df.copy()

    # Remove constant sensors
    df = df.drop(columns=CONSTANT_SENSORS)

    # Calculate Remaining Useful Life
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]

    return df