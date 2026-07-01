"""
First-order, one-zone CO2 model. This is deliberately simple: it is a
single mixed-volume approximation, not a CFD or multi-zone model, and the
pitch script should say so explicitly if a technical judge asks.

dCO2/dt = generation (occupancy) - removal (action-dependent rate)

Removal rates below are illustrative starting points, not measured values.
Make them configurable in the UI so the "trust dial" framing is honest.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTDOOR_CO2 = 420  # ppm, background atmospheric level
GENERATION_PER_OCCUPANT = 1.3  # ppm/min added to room average, tuned so an 8-person
# meeting room rising from ~910 ppm breaches 1000 ppm in ~15-20 min under do_nothing,
# matching the original dashboard mock. Recalibrate against your own preset numbers.

# removal rate constants (per minute, fraction of the gap to outdoor closed per minute)
# NOTE: standalone particulate filtration (HEPA/carbon) does not meaningfully remove
# CO2, it only exchanges or scrubs particulates/VOCs. filtration is deliberately set
# close to do_nothing here. Do not show filtration as a CO2 fix in the main forecast
# chart, that is a physics error a technical judge will likely catch; use filtration
# only in the outdoor-air-quality / PM preset (Preset C), not the CO2 breach preset.
ACTION_REMOVAL_RATE = {
    "do_nothing": 0.012,
    "boost_ventilation": 0.05,
    "open_window": 0.085,
    "filtration": 0.012,  # intentionally identical to do_nothing; see note above
}

# open_window is penalized if outdoor air quality is poor (e.g. smoke); 1.0 = no penalty
OUTDOOR_AIR_PENALTY = {
    "do_nothing": 1.0,
    "boost_ventilation": 1.0,
    "open_window": 1.0,  # multiply by e.g. 0.4 in the preset when outdoor air is poor
    "filtration": 1.0,
}


@dataclass
class CurrentState:
    co2: float
    occupant_count: int
    outdoor_air_poor: bool = False


def simulate_intervention(
    state: CurrentState,
    action: str,
    horizon_minutes: int = 30,
    co2_target: int = 1000,
    step_minutes: float = 1.0,
) -> pd.DataFrame:
    """Return a dataframe with time, co2, risk status, and threshold flag for
    one action, simulated forward from the current state."""
    if action not in ACTION_REMOVAL_RATE:
        raise ValueError(f"unknown action '{action}', expected one of {list(ACTION_REMOVAL_RATE)}")

    removal_rate = ACTION_REMOVAL_RATE[action]
    if action == "open_window" and state.outdoor_air_poor:
        removal_rate *= 0.4  # window opening is much less attractive with poor outdoor air

    steps = int(horizon_minutes / step_minutes)
    co2 = state.co2
    rows = []
    for t in range(steps + 1):
        rows.append({
            "minute": t * step_minutes,
            "co2": round(co2, 1),
            "above_threshold": co2 > co2_target,
        })
        generation = GENERATION_PER_OCCUPANT * state.occupant_count
        removal = removal_rate * (co2 - OUTDOOR_CO2)
        co2 = co2 + (generation - removal) * step_minutes
        co2 = max(co2, OUTDOOR_CO2)
    return pd.DataFrame(rows)


def minutes_above_threshold(curve: pd.DataFrame) -> float:
    return curve["above_threshold"].sum() * (curve["minute"].iloc[1] - curve["minute"].iloc[0])


if __name__ == "__main__":
    state = CurrentState(co2=910, occupant_count=8)
    for action in ACTION_REMOVAL_RATE:
        curve = simulate_intervention(state, action, horizon_minutes=30)
        print(f"{action:20s} end_co2={curve['co2'].iloc[-1]:7.1f}  "
              f"minutes_above_threshold={minutes_above_threshold(curve):.0f}")
