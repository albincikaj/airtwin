# AirTwin: Session Update

Last updated: Wednesday, July 1, 2026 (final build day, see Timeline reality below).
This file is the single source of truth for state. Read this and task.md in
full before changing anything. Update both before ending any session.

## 1. What this project is

AirTwin is a room-level CO2 forecasting and prescriptive ventilation digital
twin for the DTU hackathon. It predicts an upcoming CO2 threshold breach,
simulates candidate interventions, and now automatically decides and applies
one, closing the loop rather than only recommending.

## 2. Timeline reality

The team decision is that today, Wednesday, July 1, 2026, is the last full
build day; the two days before this were spent on research and direction
changes rather than building. Every task in task.md is scoped for a same-day
ship. Anything not marked P0 should be cut without debate if time runs short.

## 3. How we got here (decision log)

1. Original direction was manufacturing predictive maintenance (AI4I 2020
   dataset, custom tool-wear simulator). Dropped in full; not being kept as
   a fallback.
2. Pivoted to room-level IAQ and CO2 forecasting using the UCI Occupancy
   Detection dataset (https://archive.ics.uci.edu/dataset/357/occupancy+detection),
   20,560 rows, one office room, February 2 to 18, 2015, CC BY 4.0.
3. Default breach horizon set to 30 minutes, not 15. The 15-minute positive
   rate is only 1.14 percent (234 of 20,560 rows); 30 minutes gives about
   2.2 percent (roughly 450 of 20,560), a better signal to train and
   evaluate against. Both are implemented; 30 is the default.
4. Both HistGradientBoostingClassifier and RandomForestClassifier are
   trained with sample weights from class frequency (class_weight
   "balanced") to compensate for that imbalance. Unweighted training
   collapses to "always predict no breach."
5. HistGradientBoostingClassifier is the default model: F1 0.235, precision
   0.134, recall 0.943, versus RandomForestClassifier's F1 0.166, precision
   0.091, recall 0.959, on the same chronological held-out split (4,097
   rows, 122 positive). Expect high recall, low precision from either; that
   trade is a deliberate, explainable design choice, not a bug.
6. Filtration is modeled as functionally identical to do_nothing for CO2
   (same removal rate constant) because standalone particulate filtration
   does not remove CO2. It is excluded from the closed-loop action set for
   that reason and should only appear in a separate outdoor-air-quality
   scenario, not the CO2 breach story.
7. OpenAQ is cut from the plan. It now requires registration and an API
   key, and only covers CO2 for a small number of locations; not worth the
   setup time today.
8. AQUAIR is excluded from every slide, document, and source list. It is
   fish-tank aquaculture sensor data from a facility in Morocco, not an
   office or classroom IAQ dataset, despite reading generically in a list.
9. Built src/closed_loop.py to add an actual automatic control loop.
   [Assumption, open, see section 5] The course material distinguishes a
   Digital Shadow (sensor data in, a human reads a dashboard) from a
   Digital Twin (sensor data in, control decisions automatically flow back
   out and change what gets sensed next). Everything before this file was a
   well-built Shadow. This file is what makes it a Twin.

## 4. Verified numbers (reuse these, do not re-derive)

- Combined dataset: 20,560 rows, February 2, 2015 14:19 to February 18,
  2015 09:19, zero duplicate timestamps, timestamps almost exactly 60
  seconds apart (median = mean = 60.0s, max gap 61s).
- CO2 mean 606.5 ppm, median 453.5 ppm, max 2,028.5 ppm across the training
  file; 12.0 percent of training rows already sit above 1,000 ppm.
- breach_next_15min positive rate: 1.14 percent (234 of 20,560).
  breach_next_30min positive rate: about 2.2 percent (roughly 450 of
  20,560; exact count depends on which boundary rows get filtered).
- HistGradientBoostingClassifier (30-minute horizon, held-out test of 4,097
  rows, 122 positive): F1 0.235, precision 0.134, recall 0.943, confusion
  matrix [[3233, 742], [7, 115]].
- RandomForestClassifier, same split: F1 0.166, precision 0.091, recall
  0.959, confusion matrix [[2802, 1173], [5, 117]].
- Evaluation ablation from `src/evaluation.py` on the same 4,097-row
  chronological hold-out: rules-only F1 0.032, precision 0.017,
  recall 0.213, confusion matrix [[2480, 1495], [96, 26]];
  ML-only HistGradientBoosting F1 0.235, precision 0.134, recall
  0.943, confusion matrix [[3233, 742], [7, 115]]; hybrid at
  model_weight 0.50 F1 0.103, precision 0.055, recall 0.861,
  confusion matrix [[2154, 1821], [17, 105]].
- Hybrid true-positive alerts gave a median 14.0 minutes of warning
  before the actual CO2 crossing (mean 14.2, min 1.0, max 30.0,
  n=105 true positives).
- Closed-loop proof (12-person meeting, 35 minutes occupied then clearing,
  starting at 910 ppm, same physics both runs): forced do_nothing throughout
  breaches for 40 of 50 minutes, peak 1,189 ppm. Automatic closed loop
  breaches for 0 of 50 minutes, peak 964 ppm.

## 5. Open questions and blockers

1. HIGH PRIORITY, blocking the pitch framing: confirm the exact wording of
   the Digital Twin versus Digital Shadow requirement against the actual
   course lecture slide or kickoff material, not this file's paraphrase of
   an earlier conversation summary. If the real requirement demands literal
   physical actuation rather than an automatic software loop, say so
   immediately; today's scope changes.
2. Whether a physical hardware element (a smart plug or relay switching a
   small fan from closed_loop.py's action_taken) is worth attempting.
   Default answer is no unless a part is already on hand; it is P2.
3. Which setting to lock as the demo narrative. Current default is a
   meeting room. The original plan also allowed classroom, office, or
   hospital waiting room; no reason found yet to change from meeting room.
4. Edge-case review: the closed-loop proof should be demoed at the default
   hybrid weight 0.50. At model_weight 1.00, the first 15 minutes have no
   ML history, so the loop behaves like do_nothing and the 40-versus-0
   proof disappears. Zero-occupant runs are stable.

## 6. Current build status

| Component | Status | Notes |
|---|---|---|
| Data pipeline (data_loader.py, features.py) | Done, tested | Runs against real downloaded data |
| Rule baseline (rule_model.py) | Done, tested | Includes hybrid_risk |
| ML model (ml_model.py) | Done, tested | Artifacts saved in models/ |
| Simulator (simulator.py) | Done, tested, calibrated | Filtration intentionally inert for CO2 |
| Recommender (recommender.py) | Done, tested | |
| Closed loop (closed_loop.py) | Done, tested | The Digital Twin proof |
| Dashboard (app.py) | Done, tested | Presets A, B, D; styled room panel; evaluation panel auto-loads src/evaluation.py |
| Preset C (outdoor air, filtration) | Not started | P1 |
| Evaluation ablation (rules vs ML vs hybrid) | Done, tested | Implemented in src/evaluation.py; dashboard auto-loads get_eval_results() |
| Demo script | Not started | |
| Backup video | Not started | |
| GitHub push | Repo is committed locally, not yet pushed | |
| Written report | Not started | |
| Slide deck | Not started | |

## 7. Bootstrap protocol for any new session (Claude Code or Codex)

1. Read this file in full.
2. Read task.md in full and find the first unchecked task in your section.
3. Run `uv sync` then `uv run streamlit run src/app.py` (or the relevant
   module directly) to confirm the environment still works before changing
   anything.
4. State which task you are starting and confirm no assumption above has
   changed since this file was last updated.
5. On finishing a task: check it off in task.md, and if you made a
   decision that isn't already logged in section 3, add it before ending
   the session.

## 8. Changelog

- Wednesday, July 1, 2026: file created after closed_loop.py was built and
  verified; repo given its first git commit.
- Wednesday, July 1, 2026: src/evaluation.py added with rules-only,
  ML-only, and hybrid ablation metrics, hybrid lead-time summary, and a
  reusable closed-loop comparison summary.
- Wednesday, July 1, 2026 (session 2): Claude Code completed all P0 dashboard
  tasks. Room visual replaced metric cards with an HTML panel (status color,
  people icons, fan/window state). Evaluation panel added to app.py: it
  dynamically imports evaluation.get_eval_results() if available, otherwise
  falls back to the verified hardcoded numbers from section 4 of this file.
  Rough-edge pass done: st.spinner on closed-loop run, try/except around
  run_closed_loop, st.warning with icon when model artifact is missing. All
  preset numbers and simulator constants verified unchanged. Closed-loop
  40-vs-0 proof re-confirmed (shadow=40 min over 1000 ppm, peak 1189; twin=0
  min, peak 964). IDE "cannot find module" diagnostics for joblib/streamlit
  are false positives from the linter not resolving the uv venv path.
