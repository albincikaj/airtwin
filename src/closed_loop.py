"""
This file is the difference between a Digital Twin and a Digital Shadow, in
the specific sense your own course material draws that line: a Digital
Shadow is sensor data flowing in with a human reading a dashboard; a Digital
Twin is sensor data flowing in AND control decisions flowing back out,
automatically, so the decision changes what gets sensed next.

Everything in rule_model.py, ml_model.py, simulator.py, and recommender.py
so far can forecast and recommend a single fixed action for a fixed window.
Nothing yet re-decides at every step and lets that decision shape the next
reading. This file does that: it is a real feedback loop, not a one-shot
forecast, and it is what should be on screen when someone asks "where is
the closed loop" during judging.
"""
from collections import deque

import joblib
import pandas as pd

from rule_model import RoomState, hybrid_risk, rule_risk
from simulator import ACTION_REMOVAL_RATE, GENERATION_PER_OCCUPANT, OUTDOOR_CO2

# hybrid_score -> automatic action tier. This ladder IS the control policy;
# make it a config the story owner can defend, not a black box.
ESCALATION_LADDER = (
    (0.60, "open_window"),
    (0.30, "boost_ventilation"),
    (0.0, "do_nothing"),
)


def _choose_action(hybrid_score: float, outdoor_air_poor: bool) -> str:
    for cutoff, action in ESCALATION_LADDER:
        if hybrid_score >= cutoff:
            if action == "open_window" and outdoor_air_poor:
                return "boost_ventilation"  # automatic substitution, still automatic
            return action
    return "do_nothing"


def _ml_score_from_history(model, co2_history: deque, occupied: bool) -> float:
    """Rebuild the same lag/slope/rollmax features the trained model expects,
    but from the live rolling window instead of the historical CSV. If there
    is not yet enough history (first few minutes), fall back to 0.0 so the
    rule layer alone drives the early decisions."""
    if model is None or len(co2_history) < 16:
        return 0.0
    h = list(co2_history)  # oldest first
    row = {}
    for m in (1, 3, 5):
        row[f"co2_lag_{m}"] = h[-1 - m]
        row[f"temp_lag_{m}"] = 22.0
        row[f"humidity_lag_{m}"] = 30.0
        row[f"light_lag_{m}"] = 400.0
    for w in (5, 10, 15):
        row[f"co2_slope_{w}"] = (h[-1] - h[-1 - w]) / w
        row[f"co2_rollmax_{w}"] = max(h[-1 - w:])
    row.update({"Temperature": 22.0, "Humidity": 30.0, "Light": 400.0, "CO2": h[-1],
                "HumidityRatio": 0.005, "Occupancy": int(occupied),
                "hour": 14, "minute": 0, "day_segment": "afternoon"})
    from ml_model import ml_risk
    return float(ml_risk(model, pd.DataFrame([row]))[0])


def run_closed_loop(
    initial_co2: float,
    occupancy_schedule: list,
    co2_target: int = 1000,
    model=None,
    model_weight: float = 0.5,
    outdoor_air_poor: bool = False,
    force_action: str | None = None,
) -> pd.DataFrame:
    """Run the closed loop for len(occupancy_schedule) minutes.
    occupancy_schedule[t] gives (occupied: bool, occupant_count: int) for
    minute t; vary it to show the loop both escalating and de-escalating.
    Pass force_action to bypass the automatic policy and hold one action
    fixed throughout, which is how you generate an honest "no twin" /
    Digital Shadow counterfactual for the same physics, rather than relabeling
    the controlled run after the fact.
    Returns one row per minute: what was sensed, decided, and the effect."""
    co2 = initial_co2
    history = deque(maxlen=20)
    history.append(co2)
    rows = []

    for t, (occupied, occupant_count) in enumerate(occupancy_schedule):
        slope_5 = (history[-1] - history[-min(6, len(history))]) / min(5, len(history) - 1) if len(history) > 1 else 0.0

        rule_result = rule_risk(RoomState(co2=co2, co2_slope_5min=slope_5, occupied=occupied, co2_target=co2_target))
        ml_score = _ml_score_from_history(model, history, occupied)
        hybrid = hybrid_risk(rule_result["risk_score"], ml_score, model_weight)

        action = force_action if force_action else _choose_action(hybrid["hybrid_score"], outdoor_air_poor)

        rows.append({
            "minute": t, "co2_sensed": round(co2, 1), "occupied": occupied,
            "occupant_count": occupant_count, "rule_status": rule_result["status"],
            "hybrid_score": hybrid["hybrid_score"], "action_taken": action,
        })

        # ACT: apply one minute of the chosen action, this is the control
        # signal going back out and changing the plant, closing the loop.
        removal_rate = ACTION_REMOVAL_RATE[action]
        generation = GENERATION_PER_OCCUPANT * occupant_count
        removal = removal_rate * (co2 - OUTDOOR_CO2)
        co2 = max(co2 + generation - removal, OUTDOOR_CO2)
        history.append(co2)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # a busy meeting that fills up, stays full, then empties: escalation AND de-escalation
    schedule = (
        [(True, 12)] * 35   # meeting running, larger group
        + [(False, 0)] * 15  # room clears
    )

    model_path = "../models/hist_gb_breach_30min.joblib"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        model = None

    print("=== WITHOUT the loop (forced do_nothing throughout, a Digital Shadow) ===")
    shadow = run_closed_loop(910, schedule, model=model, model_weight=0.5, force_action="do_nothing")
    print(f"minutes above target: {(shadow['co2_sensed'] > 1000).sum()} of {len(shadow)}, "
          f"peak CO2: {shadow['co2_sensed'].max():.0f} ppm")

    print("\n=== WITH the loop (automatic sense-decide-act, a Digital Twin) ===")
    twin = run_closed_loop(910, schedule, model=model, model_weight=0.5)
    print(f"minutes above target: {(twin['co2_sensed'] > 1000).sum()} of {len(twin)}, "
          f"peak CO2: {twin['co2_sensed'].max():.0f} ppm")
    print(twin[["minute", "co2_sensed", "hybrid_score", "action_taken"]].to_string(index=False))

