"""
Run every candidate action through the simulator, score it, and recommend
the cheapest one that actually keeps the room under target (or minimizes
breach time if none fully avoid it).
"""
import pandas as pd

from simulator import CurrentState, minutes_above_threshold, simulate_intervention

# Subjective friction cost, 0 (free) to 1 (disruptive). These are placeholders;
# the story-owner teammate should sanity-check these against the chosen setting
# (meeting room vs classroom vs hospital waiting room changes "open window" a lot).
ACTION_FRICTION = {
    "do_nothing": 0.0,
    "boost_ventilation": 0.15,
    "open_window": 0.35,
    "filtration": 0.25,
}

# CO2 doesn't respond to filtration; only offer it as a candidate when the
# story is about particulates (Preset C), not the CO2 breach presets.
CO2_RELEVANT_ACTIONS = ("do_nothing", "boost_ventilation", "open_window")


def recommend_action(
    state: CurrentState,
    actions: tuple[str, ...] = CO2_RELEVANT_ACTIONS,
    horizon_minutes: int = 30,
    co2_target: int = 1000,
) -> dict:
    """Run simulations, score interventions, and return the best action plus
    an explanation. Lower score is better: minutes above threshold dominates,
    friction is the tiebreaker."""
    results = []
    for action in actions:
        curve = simulate_intervention(state, action, horizon_minutes, co2_target)
        minutes_over = minutes_above_threshold(curve)
        results.append({
            "action": action,
            "minutes_above_threshold": minutes_over,
            "friction": ACTION_FRICTION.get(action, 0.5),
            "end_co2": curve["co2"].iloc[-1],
            "curve": curve,
        })

    results.sort(key=lambda r: (r["minutes_above_threshold"], r["friction"]))
    best = results[0]
    do_nothing = next(r for r in results if r["action"] == "do_nothing")

    gain_minutes = do_nothing["minutes_above_threshold"] - best["minutes_above_threshold"]
    if best["action"] == "do_nothing":
        explanation = "No action keeps the room under target any better than doing nothing right now."
    elif best["minutes_above_threshold"] == 0:
        explanation = (
            f"{best['action'].replace('_', ' ')} keeps CO2 under {co2_target} ppm for the "
            f"full {horizon_minutes}-minute window, avoiding {gain_minutes:.0f} minutes "
            f"above target compared with doing nothing."
        )
    else:
        explanation = (
            f"{best['action'].replace('_', ' ')} reduces time above {co2_target} ppm by "
            f"{gain_minutes:.0f} minutes compared with doing nothing, though it does not "
            f"fully avoid a breach."
        )

    return {
        "recommended_action": best["action"],
        "explanation": explanation,
        "expected_gain_minutes": gain_minutes,
        "all_results": results,
    }


if __name__ == "__main__":
    state = CurrentState(co2=910, occupant_count=8)
    rec = recommend_action(state)
    print(rec["recommended_action"], "->", rec["explanation"])
