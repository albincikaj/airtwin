"""
Evaluation helpers for the final AirTwin demo/report.

This module deliberately does not retrain models. It evaluates the existing
model artifact against the same chronological hold-out split used elsewhere,
then adds rule-only and hybrid ablations for the same rows.
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from closed_loop import run_closed_loop
from data_loader import load_train_test_split
from features import make_features
from ml_model import ml_risk
from rule_model import RoomState, hybrid_risk, rule_risk

CO2_TARGET = 1000
HORIZON_MINUTES = 30
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hist_gb_breach_30min.joblib"


def _load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def _metric_row(label: str, y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    return {
        "label": label,
        "f1": round(f1_score(y_true, preds, zero_division=0), 3),
        "precision": round(precision_score(y_true, preds, zero_division=0), 3),
        "recall": round(recall_score(y_true, preds, zero_division=0), 3),
        "confusion_matrix": cm,
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": threshold,
    }


def _evaluation_frame() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    _, test_raw = load_train_test_split(test_fraction=0.2)
    X_test, y_test = make_features(
        test_raw,
        horizon_minutes=HORIZON_MINUTES,
        co2_target=CO2_TARGET,
    )
    return X_test, y_test, test_raw.sort_values("date").reset_index(drop=True)


def get_eval_results(model_weight: float = 0.5) -> dict:
    """Return dashboard-safe ablation metrics for rules, ML, and hybrid."""
    X_test, y_test, _ = _evaluation_frame()
    model = _load_model()

    rule_scores = np.array([
        rule_risk(RoomState(
            co2=float(row.CO2),
            co2_slope_5min=float(row.co2_slope_5),
            occupied=bool(row.Occupancy),
            co2_target=CO2_TARGET,
        ))["risk_score"]
        for row in X_test.itertuples(index=False)
    ])

    results = {
        "rules_only": _metric_row("Rules only", y_test, rule_scores),
    }

    if model is not None:
        ml_scores = ml_risk(model, X_test)
        hybrid_scores = np.array([
            hybrid_risk(float(rule_score), float(ml_score), model_weight)["hybrid_score"]
            for rule_score, ml_score in zip(rule_scores, ml_scores)
        ])
        results["ml_only"] = _metric_row("ML only (HistGradientBoosting)", y_test, ml_scores)
        results["hybrid"] = _metric_row(
            f"Hybrid risk (model weight {model_weight:.2f})",
            y_test,
            hybrid_scores,
        )
    else:
        results["ml_only"] = {
            "label": "ML only (model artifact missing)",
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_test": int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "threshold": 0.5,
        }

    return results


def get_lead_time_summary(model_weight: float = 0.5) -> dict:
    """Summarize warning time for true positives from the hybrid model."""
    X_test, y_test, test_raw = _evaluation_frame()
    model = _load_model()
    if model is None:
        return {"count": 0, "reason": "model artifact missing"}

    rule_scores = np.array([
        rule_risk(RoomState(
            co2=float(row.CO2),
            co2_slope_5min=float(row.co2_slope_5),
            occupied=bool(row.Occupancy),
            co2_target=CO2_TARGET,
        ))["risk_score"]
        for row in X_test.itertuples(index=False)
    ])
    ml_scores = ml_risk(model, X_test)
    hybrid_scores = np.array([
        hybrid_risk(float(rule_score), float(ml_score), model_weight)["hybrid_score"]
        for rule_score, ml_score in zip(rule_scores, ml_scores)
    ])
    preds = hybrid_scores >= 0.5

    # make_features drops the first max lag/slope rows: 15 minutes here.
    original_indices = np.arange(15, 15 + len(y_test))
    lead_times = []
    for original_idx, predicted, actual in zip(original_indices, preds, y_test):
        if not predicted or not actual:
            continue
        now = test_raw.loc[original_idx, "date"]
        horizon_end = now + pd.Timedelta(minutes=HORIZON_MINUTES)
        future = test_raw.loc[original_idx + 1:]
        future = future[future["date"] <= horizon_end]
        crossings = future[future["CO2"] > CO2_TARGET]
        if crossings.empty:
            continue
        first_crossing = crossings.iloc[0]["date"]
        lead_times.append((first_crossing - now).total_seconds() / 60)

    if not lead_times:
        return {"count": 0}

    values = np.array(lead_times)
    return {
        "count": int(len(values)),
        "mean_minutes": round(float(values.mean()), 1),
        "median_minutes": round(float(np.median(values)), 1),
        "min_minutes": round(float(values.min()), 1),
        "max_minutes": round(float(values.max()), 1),
    }


def get_closed_loop_summary(
    initial_co2: float = 910,
    occupants: int = 12,
    occupied_minutes: int = 35,
    clear_minutes: int = 15,
    model_weight: float = 0.5,
) -> dict:
    """Return the Digital Shadow counterfactual versus automatic control."""
    model = _load_model()
    schedule = [(True, occupants)] * occupied_minutes + [(False, 0)] * clear_minutes
    shadow = run_closed_loop(
        initial_co2,
        schedule,
        model=model,
        model_weight=model_weight,
        force_action="do_nothing",
    )
    twin = run_closed_loop(initial_co2, schedule, model=model, model_weight=model_weight)

    shadow_minutes = int((shadow["co2_sensed"] > CO2_TARGET).sum())
    twin_minutes = int((twin["co2_sensed"] > CO2_TARGET).sum())
    return {
        "shadow_minutes_above_target": shadow_minutes,
        "shadow_peak_co2": int(round(shadow["co2_sensed"].max())),
        "twin_minutes_above_target": twin_minutes,
        "twin_peak_co2": int(round(twin["co2_sensed"].max())),
        "minutes_above_target_avoided": shadow_minutes - twin_minutes,
        "n_minutes": len(schedule),
    }


def get_report_summary() -> str:
    metrics = get_eval_results()
    lead = get_lead_time_summary()
    loop = get_closed_loop_summary()
    hybrid = metrics.get("hybrid", metrics["rules_only"])
    lead_sentence = (
        f"True-positive hybrid alerts gave a median {lead['median_minutes']:.1f} minutes "
        f"of warning before the actual crossing."
        if lead.get("count", 0)
        else "Lead-time could not be computed because there were no hybrid true positives."
    )
    return (
        f"On the chronological hold-out split, the hybrid model reached "
        f"F1 {hybrid['f1']:.3f}, precision {hybrid['precision']:.3f}, and "
        f"recall {hybrid['recall']:.3f}. {lead_sentence} In the closed-loop "
        f"comparison, automatic control avoided {loop['minutes_above_target_avoided']} "
        f"minutes above {CO2_TARGET} ppm versus the forced do-nothing shadow "
        f"({loop['shadow_minutes_above_target']} to {loop['twin_minutes_above_target']} "
        f"minutes), while lowering peak CO2 from {loop['shadow_peak_co2']} ppm "
        f"to {loop['twin_peak_co2']} ppm."
    )


if __name__ == "__main__":
    print("Evaluation metrics:")
    for key, row in get_eval_results().items():
        print(f"{key}: {row}")
    print("\nLead time:")
    print(get_lead_time_summary())
    print("\nClosed-loop summary:")
    print(get_closed_loop_summary())
    print("\nReport summary:")
    print(get_report_summary())
