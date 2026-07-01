# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                        # install dependencies
uv run python data_sanity_check.py             # verify dataset before building on it
uv run python src/features.py                  # sanity-check feature/label pipeline
uv run python src/rule_model.py                # run rule baseline (no data needed)
uv run python src/simulator.py                 # run intervention curves (no data needed)
uv run python src/recommender.py               # run action recommender
uv run python src/ml_model.py                  # train classifiers → saves to models/
uv run streamlit run src/app.py                # launch dashboard
```

Run scripts in the order above the first time. `ml_model.py` must run before the dashboard has a live ML score (until then, app.py falls back to a placeholder score).

## Architecture

AirTwin is a **room-level CO2 breach forecasting and prescriptive ventilation dashboard** built on the UCI Occupancy Detection dataset (20,560 rows, Feb 2–18 2015, one office room, ~60-second sampling interval). The dataset lives in `data/raw/`; all three files (datatraining.txt, datatest.txt, datatest2.txt) are combined chronologically before any split.

### Data pipeline (`data_loader.py` → `features.py`)

`load_all_chronological()` merges and sorts all three files; `load_train_test_split()` does a strict chronological split (never shuffle — this is a time series). `make_features()` builds backward-looking lag/slope/rollmax features and a forward-looking breach label.

**Critical invariant:** input features look backward only; labels look forward only. Accidentally centering a rolling window or shifting a label the wrong direction causes silent future leakage — the model cheats during training and then fails live.

### Risk scoring — three layers

1. **Rule layer** (`rule_model.py::rule_risk`): deterministic thresholds on current CO2, 5-min slope, and occupancy. Always explainable, no trained model required.
2. **ML layer** (`ml_model.py::ml_risk`): `HistGradientBoostingClassifier` (preferred over RF on this data) trained with `compute_sample_weight("balanced")` to compensate for class imbalance (~2.2% positive at 30-min horizon). Saved to `models/hist_gb_breach_30min.joblib`.
3. **Hybrid** (`rule_model.py::hybrid_risk`): weighted blend controlled by a UI slider. `model_weight=0` is rules only, `model_weight=1` is ML only — treat it as a trust dial, not a principled coefficient.

### Simulator and recommender (`simulator.py`, `recommender.py`)

`simulate_intervention()` is a first-order single-zone CO2 mass-balance model: `dCO2/dt = generation − removal`. `recommend_action()` runs every candidate action through the simulator and picks the one minimizing minutes above threshold, with friction as a tiebreaker.

**Filtration physics:** standalone HEPA/carbon filtration does not remove CO2. `filtration` is intentionally set equal to `do_nothing` in `ACTION_REMOVAL_RATE`. Do not show filtration as a CO2 fix; use it only for particulate presets (Preset C).

### Dashboard (`src/app.py`)

Streamlit app with three hardcoded presets (A, B, D). Requires three Streamlit patterns to work correctly across interactions:
- `st.cache_data` — dataset loading (expensive, never changes mid-demo)
- `st.cache_resource` — model loading (avoids reloading on every slider drag)
- `st.session_state` — active preset + slider value (survives full script reruns)

## Key constants to recalibrate before demo

- `simulator.py`: `GENERATION_PER_OCCUPANT` (1.3 ppm/min), `ACTION_REMOVAL_RATE` dict
- `recommender.py`: `ACTION_FRICTION` dict (subjective; depends on room setting)
- `app.py`: CO2/slope/occupant values in the `PRESETS` dict

## Known limits to disclose proactively

- Occupancy is binary (not a headcount); "N people est." on the dashboard is a heuristic.
- One office room, ~2 weeks in Feb 2015 — not a live feed.
- Simulator is a single-zone approximation, not CFD.
- Expected ML metrics: ~94–96% recall, ~9–13% precision — intentional high-recall tuning, not a bug.
