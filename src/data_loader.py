"""
Load the UCI Occupancy Detection dataset.
Source: https://archive.ics.uci.edu/dataset/357/occupancy+detection

Three files are provided (datatraining.txt, datatest.txt, datatest2.txt).
They are pre-split, but datatraining.txt alone only contains ~67 breach
events out of 8143 rows at a 15-minute horizon (verified: see
data_sanity_check.py). Combining all three chronologically before re-splitting
gives more positive examples to both train and evaluate on, while still
keeping the test portion strictly later in time than the train portion.
"""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REQUIRED_COLUMNS = ["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]


def _load_one(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download from "
            f"https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip "
            f"and unzip into data/raw/"
        )
    df = pd.read_csv(path, index_col=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing expected columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df[REQUIRED_COLUMNS]


def load_all_chronological() -> pd.DataFrame:
    """All three files combined and sorted by time (20,560 rows total).
    Use this for EDA and feature engineering."""
    frames = [_load_one(f) for f in ("datatraining.txt", "datatest.txt", "datatest2.txt")]
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return full


def load_train_test_split(test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """True chronological split: the earliest (1 - test_fraction) of rows are
    train, the most recent test_fraction are test. Never shuffle this; shuffling
    a time series before splitting leaks future rows into training."""
    full = load_all_chronological()
    cutoff = int(len(full) * (1 - test_fraction))
    return full.iloc[:cutoff].copy(), full.iloc[cutoff:].copy()


if __name__ == "__main__":
    full = load_all_chronological()
    print(f"loaded {len(full)} rows, {full['date'].min()} to {full['date'].max()}")
    print(f"occupied rate: {full['Occupancy'].mean() * 100:.1f}%")
    print(f"CO2 mean/max: {full['CO2'].mean():.0f} / {full['CO2'].max():.0f} ppm")
