"""
Build features and breach labels from the raw occupancy series.

Two different time directions matter here, and mixing them up is the most
common bug in this kind of pipeline:
  - INPUT FEATURES must only look backward (lags, rolling stats up to now).
  - LABELS must look forward (did a breach happen in the next N minutes).
If a rolling window is accidentally centered, or a label is shifted the
wrong way, the model quietly "cheats" with future information during
training and then fails live, which is much worse than an honest low score.

Verified against the real data (data_sanity_check.py): timestamps are
almost exactly 60 seconds apart (median = mean = 60.0s, max gap 61s), so
row-based lags are a safe stand-in for time-based lags here.
"""
import pandas as pd

LAG_MINUTES = (1, 3, 5)
SLOPE_WINDOWS = (5, 10, 15)


def make_features(
    df: pd.DataFrame,
    horizon_minutes: int = 15,
    co2_target: int = 1000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature dataframe X and label series y for future CO2 breach.

    y = 1 if CO2 is currently at or below co2_target AND crosses above it at
    some point in the next `horizon_minutes`. Rows already above the target
    are excluded from the positive class on purpose (the rule layer already
    flags "currently critical"; this label is specifically the early warning).
    """
    out = df.copy().sort_values("date").reset_index(drop=True)

    # --- backward-looking input features ---
    for m in LAG_MINUTES:
        out[f"co2_lag_{m}"] = out["CO2"].shift(m)
        out[f"temp_lag_{m}"] = out["Temperature"].shift(m)
        out[f"humidity_lag_{m}"] = out["Humidity"].shift(m)
        out[f"light_lag_{m}"] = out["Light"].shift(m)

    for w in SLOPE_WINDOWS:
        # ppm change per minute over the trailing window; backward-looking only
        out[f"co2_slope_{w}"] = (out["CO2"] - out["CO2"].shift(w)) / w
        out[f"co2_rollmax_{w}"] = out["CO2"].rolling(w, min_periods=1).max()

    out["hour"] = out["date"].dt.hour
    out["minute"] = out["date"].dt.minute
    out["day_segment"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 21, 24],
        labels=["night", "morning", "afternoon", "evening", "night_late"],
    ).astype(str)

    # --- forward-looking label (time-indexed reversed rolling max) ---
    co2_by_time = out.set_index("date")["CO2"]
    reversed_series = co2_by_time.iloc[::-1]
    forward_max = reversed_series.rolling(f"{horizon_minutes}min").max().iloc[::-1]
    forward_max = forward_max.reset_index(drop=True)

    currently_under = out["CO2"] <= co2_target
    y = ((forward_max > co2_target) & currently_under).astype(int)
    y.name = f"breach_next_{horizon_minutes}min"

    feature_cols = [c for c in out.columns if c.startswith(("co2_lag", "temp_lag",
                     "humidity_lag", "light_lag", "co2_slope", "co2_rollmax"))]
    feature_cols += ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio",
                      "Occupancy", "hour", "minute", "day_segment"]

    X = out[feature_cols]

    # drop rows where lag/slope features are NaN (start of series only)
    valid = X[[c for c in feature_cols if "lag" in c or "slope" in c]].notna().all(axis=1)
    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import load_all_chronological

    full = load_all_chronological()
    for h in (15, 30):
        X, y = make_features(full, horizon_minutes=h)
        print(f"horizon={h}min: {len(X)} usable rows, {y.sum()} positive "
              f"({y.mean() * 100:.2f}%)")
