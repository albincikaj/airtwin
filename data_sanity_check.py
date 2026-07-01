"""
Run this FIRST, before writing any features.
Answers the question: at the 1000 ppm threshold, how often does this room
actually breach, and what does the sensor's time spacing look like?
If breaches are very rare, you have a class-imbalance problem on day one.
"""
import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    train = load_raw("data/raw/datatraining.txt")

    print("=== Timestamp spacing ===")
    deltas = train["date"].diff().dropna().dt.total_seconds()
    print(f"median gap: {deltas.median():.1f}s | mean gap: {deltas.mean():.1f}s "
          f"| max gap: {deltas.max():.1f}s")

    print("\n=== CO2 distribution (training file, 8143 rows) ===")
    print(train["CO2"].describe().round(1).to_string())

    for target in (800, 900, 1000, 1100, 1200):
        pct_above = (train["CO2"] > target).mean() * 100
        print(f"rows with CO2 > {target} ppm: {pct_above:.1f}%")

    print("\n=== breach_next_15min @ 1000 ppm threshold ===")
    # forward-looking: does CO2 cross the target at any point in the next 15 min?
    # implemented as a time-indexed reversed rolling max, then reversed back,
    # so it only looks FORWARD from each row (labels may peek ahead, features must not)
    s = train.set_index("date")["CO2"]
    reversed_s = s.iloc[::-1]
    fwd_max_15 = reversed_s.rolling("15min").max().iloc[::-1]
    breach_15 = (fwd_max_15 > 1000) & (s <= 1000)  # not already breached right now
    print(f"positive rate (about to breach within 15 min, currently under): "
          f"{breach_15.mean() * 100:.1f}%  ({breach_15.sum()} of {len(breach_15)} rows)")

    fwd_max_30 = reversed_s.rolling("30min").max().iloc[::-1]
    breach_30 = (fwd_max_30 > 1000) & (s <= 1000)
    print(f"positive rate (about to breach within 30 min, currently under): "
          f"{breach_30.mean() * 100:.1f}%  ({breach_30.sum()} of {len(breach_30)} rows)")
