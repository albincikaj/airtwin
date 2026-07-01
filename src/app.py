"""
Streamlit dashboard for AirTwin.

Caching and session_state rationale (do not remove these):
- st.cache_data: reruns ONLY if arguments change. Used for dataset loading,
  which is expensive and never changes mid-demo.
- st.cache_resource: like cache_data but objects are not copied on rerun.
  Used for the trained model so slider drags don't reload it.
- st.session_state: survives Streamlit's full-script rerun on every
  interaction. Without it, clicking a preset then dragging the slider would
  forget which preset was chosen.
"""
import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_all_chronological
from closed_loop import run_closed_loop
from ml_model import ml_risk
from recommender import recommend_action
from rule_model import RoomState, hybrid_risk, rule_risk
from simulator import CurrentState, simulate_intervention

st.set_page_config(page_title="AirTwin", layout="wide")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hist_gb_breach_30min.joblib"

PRESETS = {
    "A: meeting room fills up": {"co2": 910, "slope": 4.2, "occupied": True, "occupants": 8},
    "B: room clears, slow decay": {"co2": 850, "slope": -0.4, "occupied": False, "occupants": 0},
    "D: do-nothing breaches": {"co2": 960, "slope": 5.5, "occupied": True, "occupants": 10},
}

# Status → colors used in the room visual panel
_STATUS_CFG = {
    "normal":    {"bg": "#d5f5e3", "border": "#1e8449", "text": "#1e8449", "badge": "NORMAL",     "badge_bg": "#1e8449"},
    "recovering":{"bg": "#d1f2eb", "border": "#17a589", "text": "#148f77", "badge": "RECOVERING", "badge_bg": "#17a589"},
    "warning":   {"bg": "#fef9e7", "border": "#d68910", "text": "#b7770d", "badge": "⚠ WARNING",  "badge_bg": "#d68910"},
    "critical":  {"bg": "#fdedec", "border": "#e74c3c", "text": "#c0392b", "badge": "🚨 CRITICAL","badge_bg": "#e74c3c"},
}


def _room_panel_html(co2: float, status: str, occupants: int, occupied: bool,
                     action: str, slope: float) -> str:
    cfg = _STATUS_CFG.get(status, _STATUS_CFG["normal"])

    # People icons — cap at 12 for visual clarity
    n = min(occupants, 12)
    if n > 0:
        row1 = "👤 " * min(n, 6)
        row2 = ("👤 " * max(0, n - 6)) if n > 6 else ""
        people_html = row1.strip()
        if row2:
            people_html += f"<br>{row2.strip()}"
    else:
        people_html = "<em style='color:#aaa;font-size:13px'>No occupants</em>"

    # Slope trend indicator
    if slope > 1.0:
        trend_html = f"<span style='color:#e74c3c'>↑ +{slope:.1f} ppm/min</span>"
    elif slope < -1.0:
        trend_html = f"<span style='color:#1e8449'>↓ {slope:.1f} ppm/min</span>"
    else:
        trend_html = f"<span style='color:#888'>→ {slope:+.1f} ppm/min</span>"

    # Fan and window — always show both; dim whichever is inactive
    fan_active = action == "boost_ventilation"
    win_active = action == "open_window"
    fan_opacity = "1" if fan_active else "0.22"
    win_opacity = "1" if win_active else "0.22"
    fan_label_color = cfg["text"] if fan_active else "#bbb"
    win_label_color = cfg["text"] if win_active else "#bbb"
    fan_label = "HVAC BOOSTED" if fan_active else "hvac idle"
    win_label = "WINDOW OPEN" if win_active else "window closed"

    return f"""
    <div style="
        background:{cfg['bg']};border:3px solid {cfg['border']};border-radius:14px;
        padding:28px 24px 20px 24px;text-align:center;font-family:sans-serif;
    ">
        <div style="font-size:11px;letter-spacing:2px;color:#999;margin-bottom:4px;
                    text-transform:uppercase">CO₂ Level</div>
        <div style="font-size:58px;font-weight:900;color:{cfg['text']};line-height:1.05;">
            {co2:.0f}<span style="font-size:22px;font-weight:400"> ppm</span>
        </div>
        <div style="font-size:13px;margin-top:4px">{trend_html}</div>
        <div style="
            display:inline-block;background:{cfg['badge_bg']};color:white;
            border-radius:20px;padding:4px 20px;font-size:12px;font-weight:700;
            margin-top:10px;letter-spacing:1.5px;
        ">{cfg['badge']}</div>
        <div style="margin-top:18px;font-size:22px;letter-spacing:4px;line-height:1.7;">
            {people_html}
        </div>
        <div style="font-size:12px;color:#777;margin-top:6px">
            {'Occupied' if occupied else 'Unoccupied'} &nbsp;·&nbsp; {occupants} est.
        </div>
        <div style="display:flex;justify-content:center;gap:32px;margin-top:18px;">
            <div style="text-align:center;opacity:{fan_opacity}">
                <div style="font-size:28px">🌀</div>
                <div style="font-size:10px;color:{fan_label_color};font-weight:600;
                            letter-spacing:0.5px;margin-top:2px">{fan_label}</div>
            </div>
            <div style="text-align:center;opacity:{win_opacity}">
                <div style="font-size:28px">🪟</div>
                <div style="font-size:10px;color:{win_label_color};font-weight:600;
                            letter-spacing:0.5px;margin-top:2px">{win_label}</div>
            </div>
        </div>
        <div style="font-size:10px;color:#bbb;margin-top:14px;">target: 1,000 ppm</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_all_chronological()


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "active_preset" not in st.session_state:
    st.session_state.active_preset = "A: meeting room fills up"
if "model_weight" not in st.session_state:
    st.session_state.model_weight = 0.5

# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------
st.title("AirTwin: room-level CO₂ forecast and prescriptive ventilation")

col_buttons, col_slider = st.columns([2, 1])
with col_buttons:
    for name in PRESETS:
        if st.button(name, key=f"preset_{name}"):
            st.session_state.active_preset = name
with col_slider:
    st.session_state.model_weight = st.slider(
        "Hybrid weight (0 = rules only, 1 = model only)",
        0.0, 1.0, st.session_state.model_weight, key="hybrid_slider",
    )

# ---------------------------------------------------------------------------
# Compute risk scores for active preset
# ---------------------------------------------------------------------------
preset = PRESETS[st.session_state.active_preset]
state = RoomState(co2=preset["co2"], co2_slope_5min=preset["slope"], occupied=preset["occupied"])
rule_result = rule_risk(state)

model = get_model()
_model_missing = model is None
if model is not None:
    feature_row = pd.DataFrame([{
        **{f"co2_lag_{m}": preset["co2"] for m in (1, 3, 5)},
        **{f"temp_lag_{m}": 22.0 for m in (1, 3, 5)},
        **{f"humidity_lag_{m}": 30.0 for m in (1, 3, 5)},
        **{f"light_lag_{m}": 400.0 for m in (1, 3, 5)},
        **{f"co2_slope_{w}": preset["slope"] for w in (5, 10, 15)},
        **{f"co2_rollmax_{w}": preset["co2"] for w in (5, 10, 15)},
        "Temperature": 22.0, "Humidity": 30.0, "Light": 400.0, "CO2": preset["co2"],
        "HumidityRatio": 0.005, "Occupancy": int(preset["occupied"]),
        "hour": 14, "minute": 0, "day_segment": "afternoon",
    }])
    ml_score = float(ml_risk(model, feature_row)[0])
else:
    ml_score = 0.9 if preset["slope"] > 0 else 0.1

hybrid = hybrid_risk(rule_result["risk_score"], ml_score, st.session_state.model_weight)

sim_state = CurrentState(co2=preset["co2"], occupant_count=preset["occupants"])
rec = recommend_action(sim_state)

# ---------------------------------------------------------------------------
# Room visual panel + risk sidebar
# ---------------------------------------------------------------------------
if _model_missing:
    st.warning(
        "No trained model found at `models/hist_gb_breach_30min.joblib`. "
        "Run `uv run python src/ml_model.py` once, then reload. "
        "ML score is a placeholder; rule score and simulator are unaffected.",
        icon="⚠️",
    )

col_room, col_risk = st.columns([2, 1])
with col_room:
    st.markdown(
        _room_panel_html(
            co2=preset["co2"],
            status=rule_result["status"],
            occupants=preset["occupants"],
            occupied=preset["occupied"],
            action=rec["recommended_action"],
            slope=preset["slope"],
        ),
        unsafe_allow_html=True,
    )
    st.caption(rule_result["explanation"])

with col_risk:
    st.metric("Hybrid risk score", f"{hybrid['hybrid_score']:.2f}",
              help="Weighted blend of rule score and ML probability")
    st.metric("Rule score", f"{hybrid['rule_score']:.2f}",
              help="Deterministic threshold-based score (always available)")
    st.metric("ML score", f"{hybrid['ml_score']:.2f}" + (" ⚠ placeholder" if _model_missing else ""),
              help="HistGradientBoostingClassifier 30-min breach probability")
    st.metric("Recommended action", rec["recommended_action"].replace("_", " "))
    st.caption(rec["explanation"])

with st.expander("Hybrid score breakdown"):
    st.json(hybrid)

# ---------------------------------------------------------------------------
# Forecast: do nothing vs recommended action
# ---------------------------------------------------------------------------
st.subheader("Forecast: do nothing vs recommended action")
do_nothing_curve = simulate_intervention(sim_state, "do_nothing", horizon_minutes=30)
best_curve = simulate_intervention(sim_state, rec["recommended_action"], horizon_minutes=30)
chart_df = pd.DataFrame({
    "minute": do_nothing_curve["minute"],
    "do_nothing": do_nothing_curve["co2"],
    rec["recommended_action"]: best_curve["co2"],
})
st.line_chart(chart_df.set_index("minute"))

# ---------------------------------------------------------------------------
# Closed-loop: Digital Twin vs Digital Shadow
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Closed loop: Digital Twin vs Digital Shadow")
st.caption(
    "A dashboard that only shows a recommendation is a Digital Shadow: sensor "
    "data in, a human decides. This section is the actual closed loop: the "
    "system senses, decides, and acts on its own every minute, and that "
    "action changes what gets sensed the next minute."
)
occupants_loop = st.slider("Meeting size for this run", 4, 16, 12, key="loop_occupants")
duration_loop = st.slider("Minutes occupied before the room clears", 10, 60, 35, key="loop_duration")

if st.button("Run closed-loop comparison"):
    schedule = [(True, occupants_loop)] * duration_loop + [(False, 0)] * 15
    with st.spinner("Running simulation…"):
        try:
            shadow = run_closed_loop(910, schedule, model=model,
                                     model_weight=st.session_state.model_weight,
                                     force_action="do_nothing")
            twin = run_closed_loop(910, schedule, model=model,
                                   model_weight=st.session_state.model_weight)
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            st.stop()

    left, right = st.columns(2)
    left.metric("Digital Shadow: minutes over target",
                int((shadow["co2_sensed"] > 1000).sum()),
                f"peak {shadow['co2_sensed'].max():.0f} ppm")
    right.metric("Digital Twin: minutes over target",
                 int((twin["co2_sensed"] > 1000).sum()),
                 f"peak {twin['co2_sensed'].max():.0f} ppm")

    compare_df = pd.DataFrame({
        "minute": shadow["minute"],
        "digital_shadow (no control)": shadow["co2_sensed"],
        "digital_twin (auto control)": twin["co2_sensed"],
    }).set_index("minute")
    st.line_chart(compare_df)
    st.dataframe(twin[["minute", "co2_sensed", "hybrid_score", "action_taken"]],
                 use_container_width=True)

# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Model evaluation (30-min breach horizon, chronological hold-out)")

# Try to load live results from evaluation.py (Codex scope).
# Falls back to verified numbers from Session_update.md §4 if not ready yet.
_eval_live = False
_eval_results = None
try:
    _eval_mod = importlib.import_module("evaluation")
    if hasattr(_eval_mod, "get_eval_results"):
        _eval_results = _eval_mod.get_eval_results()
        _eval_live = True
except Exception:
    pass

if not _eval_live:
    st.info(
        "src/evaluation.py not available yet — showing verified numbers from the "
        "training run (Session_update.md §4). This panel will auto-update once "
        "Codex's evaluation.py is in place.",
        icon="ℹ️",
    )
    # Verified numbers: 4,097-row chronological test split, 122 positives
    _eval_results = {
        "hist_gb": {
            "label": "HistGradientBoosting (default model)",
            "f1": 0.235, "precision": 0.134, "recall": 0.943,
            "confusion_matrix": [[3233, 742], [7, 115]],
            "n_test": 4097, "n_positive": 122,
        },
        "random_forest": {
            "label": "RandomForest",
            "f1": 0.166, "precision": 0.091, "recall": 0.959,
            "confusion_matrix": [[2802, 1173], [5, 117]],
            "n_test": 4097, "n_positive": 122,
        },
    }

for _key, _res in _eval_results.items():
    with st.expander(_res["label"], expanded=(_key == "hist_gb")):
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Recall", f"{_res['recall']:.3f}",
                   help="Fraction of real breaches caught — tuned high deliberately")
        _c2.metric("Precision", f"{_res['precision']:.3f}",
                   help="Fraction of alarms that were real breaches")
        _c3.metric("F1", f"{_res['f1']:.3f}")
        _cm = _res["confusion_matrix"]
        st.caption(
            f"Test set: {_res['n_test']} rows, {_res['n_positive']} positive. "
            f"Confusion matrix — TN {_cm[0][0]}, FP {_cm[0][1]}, FN {_cm[1][0]}, TP {_cm[1][1]}. "
            "High recall / low precision is the deliberate trade: catching every real breach "
            "costs some false alarms, which is the defensible framing for a safety application."
        )
