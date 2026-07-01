"""
Short-horizon breach classifier.

The positive class is rare (verified on real data: about 1.1% of rows at a
15-minute horizon, about 2.2% at 30 minutes). An unweighted model will often
just learn to always predict "no breach" and still look accurate. Sample
weights computed from class frequency are used below so rare positives count
for more during training, regardless of which of the three candidate
classifiers ends up fastest to train.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, columns=["day_segment"], drop_first=True)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, kind: str = "hist_gb"):
    Xe = _encode(X_train)
    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    if kind == "random_forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    else:
        model = HistGradientBoostingClassifier(max_depth=6, random_state=42)

    model.fit(Xe, y_train, sample_weight=weights)
    model.feature_names_ = list(Xe.columns)  # remember training-time columns
    return model


def ml_risk(model, feature_row: pd.DataFrame) -> float:
    """Return probability of CO2 threshold breach within the selected horizon
    for a single-row (or batch) feature frame."""
    Xe = _encode(feature_row)
    Xe = Xe.reindex(columns=model.feature_names_, fill_value=0)
    return model.predict_proba(Xe)[:, 1]


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    probs = ml_risk(model, X_test)
    preds = (probs >= threshold).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    return {
        "f1": round(f1_score(y_test, preds, zero_division=0), 3),
        "precision": round(report["1"]["precision"], 3) if "1" in report else 0.0,
        "recall": round(report["1"]["recall"], 3) if "1" in report else 0.0,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "n_positive_in_test": int(y_test.sum()),
        "n_test": len(y_test),
    }


if __name__ == "__main__":
    from data_loader import load_train_test_split
    from features import make_features

    train_raw, test_raw = load_train_test_split(test_fraction=0.2)
    horizon = 30  # default to 30 min: ~2x the positive examples of the 15 min label
    X_train, y_train = make_features(train_raw, horizon_minutes=horizon)
    X_test, y_test = make_features(test_raw, horizon_minutes=horizon)

    print(f"train: {len(X_train)} rows, {y_train.sum()} positive | "
          f"test: {len(X_test)} rows, {y_test.sum()} positive")

    if y_test.sum() < 5:
        print("WARNING: fewer than 5 positive examples in the test split. "
              "Metrics below will be noisy; consider a larger test_fraction "
              "or the 30-minute horizon instead of 15.")

    for kind in ("hist_gb", "random_forest"):
        model = train_model(X_train, y_train, kind=kind)
        metrics = evaluate(model, X_test, y_test)
        print(f"\n{kind}: {metrics}")
        joblib.dump(model, f"../models/{kind}_breach_{horizon}min.joblib")
