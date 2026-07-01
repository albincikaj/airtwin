# AirTwin

Room-level CO2 forecasting and prescriptive ventilation twin. Built against
the UCI Occupancy Detection dataset (https://archive.ics.uci.edu/dataset/357/occupancy+detection).

## Setup (Ubuntu)

```
cd airtwin
uv sync
```

Data is already downloaded into `data/raw/` (datatraining.txt, datatest.txt,
datatest2.txt). If you need to re-fetch it:

```
curl -L -o data/raw/occupancy.zip https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip
cd data/raw && unzip -o occupancy.zip && cd ../..
```

## Run order

```
uv run python data_sanity_check.py    # confirms class balance before you build on top of it
uv run python src/features.py         # sanity-checks the feature/label pipeline
uv run python src/rule_model.py       # rule baseline, no data needed
uv run python src/simulator.py        # intervention curves, no data needed
uv run python src/recommender.py      # picks the best action for a given state
uv run python src/ml_model.py         # trains and evaluates both classifiers, saves to models/
uv run python src/closed_loop.py      # the actual closed loop, Digital Twin vs Digital Shadow
uv run streamlit run src/app.py       # the dashboard
```

## Digital Twin vs Digital Shadow: read this before anything else

Your course material distinguishes a Digital Shadow (sensor data in, a human
reads a dashboard) from a Digital Twin (sensor data in, control decisions
flow back out automatically, and that action changes what gets sensed next).
Everything built before `closed_loop.py` forecasts and recommends, but
nothing acts on its own. `src/closed_loop.py` is the actual closed loop:
it senses CO2 every simulated minute, decides an action from the hybrid
score automatically (no human in the loop), applies that action's effect,
and re-senses. Verified on real numbers: the same starting conditions,
forced to do_nothing throughout, breach for 40 of 50 minutes (peak 1,189
ppm); with the automatic loop running, 0 of 50 minutes breach (peak 964
ppm). That contrast, not the forecast chart alone, is your strongest answer
to "where is the closed loop" in judging. It's wired into `app.py` under
"Closed loop: Digital Twin vs Digital Shadow."

[Assumption] I'm working from a summary of your course's Digital
Twin/Digital Shadow definition from an earlier conversation, not the
original lecture slide. Re-confirm the exact wording against your own
materials before you build the pitch slide around it; the concept above is
very likely right, but the precise phrasing might not be word for word.

If you want a literal hardware element on top of this (mentioned as a
"physical bonus" in your earlier planning, not a strict requirement), a
smart plug or relay switching a small fan based on `action_taken` from
`closed_loop.py` would be the fastest add, but only chase this if the
software loop above is solid first; it demonstrates the same principle
without needing hardware to work live on stage.

## What's already verified against the real data

- 20,560 total rows across the three files combined, Feb 2 to Feb 18, 2015,
  no duplicate timestamps.
- Timestamps are almost exactly 60 seconds apart (median = mean = 60.0s),
  so row-based lag features are a safe stand-in for time-based ones here.
- At a 1,000 ppm threshold, the room is already above target 12% of the
  time, so the threshold is reachable and not absurdly high or low for
  this room's data.
- The breach_next_15min label is rare: about 1.1% positive (234 of 20,560
  rows). breach_next_30min is about 2.2% positive (451 to 459 of 20,560,
  depending on how boundary rows are filtered). Both classifiers in
  ml_model.py use sample weights to compensate; expect high recall
  (roughly 94 to 96%) and low precision (roughly 9 to 13%) as the honest
  result of that trade, not a bug. Lead with this in the pitch: "we
  deliberately tuned for catching every real breach, which costs us some
  false alarms" is a defensible, judge-proof framing.
- HistGradientBoostingClassifier outperformed RandomForestClassifier
  slightly on this data (better F1 and precision at similar recall);
  default to it unless your own run says otherwise.

## What still needs tuning before the demo, not after

- `simulator.py`'s GENERATION_PER_OCCUPANT and ACTION_REMOVAL_RATE constants
  are starting points calibrated to roughly match the original dashboard
  mock (910 ppm, 8 people, breach inside 30 minutes under do_nothing,
  boost_ventilation avoiding it). Recalibrate them against whatever exact
  numbers your chosen presets use.
- `filtration` is deliberately set to remove CO2 at the same rate as
  do_nothing, because standalone particulate filtration does not remove
  CO2. Do not show it as a CO2 fix in the main forecast chart; use it only
  in the outdoor-air-quality preset (Preset C), where the tradeoff is
  particulates, not CO2.
- `app.py`'s ml_score is a placeholder until `ml_model.py` has been run and
  a model artifact exists in `models/`; swap `get_model()` to
  `joblib.load(...)` at that point.

## Known dataset limits to say out loud if asked, not hide

- Occupancy in this dataset is binary (occupied or not), not a headcount.
  Any "N people est." shown on a card is a heuristic, not a sensor reading.
- This is one office room over roughly two weeks in February 2015, not a
  live feed. Say so in the pitch; do not imply real-time sensing you don't
  have.
- The CO2 mass-balance model in `simulator.py` is a single-zone
  approximation, not a CFD model. Say this proactively if a technical
  judge asks how the curves are generated.
