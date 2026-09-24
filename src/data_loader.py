from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmapss"
    / "train_FD001.txt"
)


def load_cmapss_fd001():
    columns = [
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df