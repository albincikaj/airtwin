"""
Deterministic rule baseline. No training, no tuning, just thresholds and
slope direction. This is the layer that still works even if the ML model
has a bad day, and it's what makes every alert explainable.
"""
from dataclasses import dataclass


@dataclass
class RoomState:
    co2: float
    co2_slope_5min: float  # ppm per minute, from co2_slope_5 in features.py
    occupied: bool
    co2_target: int = 1000


def rule_risk(state: RoomState) -> dict:
    """Return {risk_score, status, explanation} using slope, occupancy, and
    current CO2. risk_score is 0 to 1."""
    rising_fast = state.co2_slope_5min > 3.0  # more than 3 ppm/min sustained rise
    decaying = state.co2_slope_5min < -1.0

    if state.co2 >= state.co2_target:
        return {
            "risk_score": 1.0,
            "status": "critical",
            "explanation": (
                f"CO2 is {state.co2:.0f} ppm, already at or above the "
                f"{state.co2_target} ppm target."
            ),
        }
    if state.occupied and rising_fast:
        # scale risk by how close we are to target and how fast we're closing the gap
        headroom = state.co2_target - state.co2
        minutes_to_breach = headroom / state.co2_slope_5min if state.co2_slope_5min > 0 else float("inf")
        urgency = max(0.0, min(1.0, 1 - (minutes_to_breach / 30)))
        return {
            "risk_score": round(0.5 + 0.4 * urgency, 2),
            "status": "warning",
            "explanation": (
                f"CO2 is {state.co2:.0f} ppm and rising about "
                f"{state.co2_slope_5min:.1f} ppm/min while occupied; "
                f"roughly {minutes_to_breach:.0f} min to target at this rate."
            ),
        }
    if (not state.occupied) and decaying:
        return {
            "risk_score": 0.05,
            "status": "recovering",
            "explanation": f"Room is unoccupied and CO2 is falling "
                            f"({state.co2_slope_5min:.1f} ppm/min).",
        }
    return {
        "risk_score": 0.2,
        "status": "normal",
        "explanation": f"CO2 is {state.co2:.0f} ppm, below target, no fast rise detected.",
    }


def hybrid_risk(rule_score: float, ml_score: float, model_weight: float = 0.5) -> dict:
    """Weighted hybrid score. model_weight is adjustable in the UI; treat it
    as a trust dial, not a principled coefficient. 0 = rules only, 1 = model only."""
    model_weight = max(0.0, min(1.0, model_weight))
    blended = model_weight * ml_score + (1 - model_weight) * rule_score
    return {
        "hybrid_score": round(blended, 3),
        "rule_score": round(rule_score, 3),
        "ml_score": round(ml_score, 3),
        "model_weight": model_weight,
    }


if __name__ == "__main__":
    print(rule_risk(RoomState(co2=910, co2_slope_5min=4.2, occupied=True)))
    print(rule_risk(RoomState(co2=1050, co2_slope_5min=1.0, occupied=True)))
    print(rule_risk(RoomState(co2=500, co2_slope_5min=-2.0, occupied=False)))
    print(hybrid_risk(rule_score=0.61, ml_score=0.94, model_weight=0.5))
