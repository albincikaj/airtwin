# Team runbook: what each person does right now

The repo and the six core functions (rule_risk, ml_risk, hybrid_risk,
simulate_intervention, recommend_action, make_features) already exist with
working starter code and real sample output below. Nobody needs to wait for
a "finished" version; everyone can start from the actual shapes shown here.

## Teammate 1: data and evaluation

Start now, no dependency on anyone else:
1. Open `data_sanity_check.py`'s output (already run, numbers in README).
   Write the one-paragraph dataset description for the report/slides using
   those numbers: 20,560 rows, Feb 2 to Feb 18, 2015, one office room.
2. Once `uv run python src/ml_model.py` has been run (ask the technical
   lead when), copy its printed F1/precision/recall/confusion matrix into
   the metrics table. The honest framing is already written for you in the
   README under "what's already verified."
3. Build the 3 demo-preset validation table by hand: for each preset, the
   columns are CO2, occupant count, expected recommended action. Compare
   against what `recommend_action()` actually returns once you can run it.

## Teammate 2: UI and visual layer

Start now, against the hardcoded example below, no need to wait:
```
rule_result = {"risk_score": 0.61, "status": "warning",
               "explanation": "CO2 is 910 ppm and rising about 4.2 ppm/min while occupied; roughly 21 min to target at this rate."}
recommendation = {"recommended_action": "boost_ventilation",
                   "explanation": "boost ventilation keeps CO2 under 1000 ppm for the full 30-minute window, avoiding 8 minutes above target compared with doing nothing."}
```
These are the real shapes the backend returns, not placeholders that will
change structure later. Build the room SVG, the four status cards, and the
forecast chart against these dictionaries directly. When the real app.py is
ready, the values update, the shape does not.

## Teammate 3: story and research grounding

Start now:
1. Your headline claim just changed, in your favor: the course distinguishes
   a Digital Shadow (dashboard, human decides) from a Digital Twin (automatic
   decision, closes the loop). Most teams will build a Shadow and call it a
   Twin. `closed_loop.py` proves yours is the real thing: same starting
   conditions, forced no-action breaches for 40 of 50 minutes; automatic
   control breaches for 0 of 50. Lead the pitch with that number.
2. [Assumption] that framing is from an earlier conversation summarizing your
   course material, not the original slide. Confirm the exact wording against
   your own lecture notes before it goes on a slide.
3. The pitch already has a working draft thread in `README.md`'s last
   section ("known dataset limits to say out loud if asked"). Turn that
   into the "do not say" slide and the proactive-disclosure line in the
   script.
4. Use ASHRAE Standard 241 and the CDC/NIOSH ventilation page only at the
   level of "this is why ventilation-based interventions are a recognized
   public-health lever," not specific clauses; the full ASHRAE 241 text is
   paywalled.
5. Do not cite the AQUAIR dataset anywhere; it is fish-tank sensor data
   from Morocco, not an office or classroom dataset, despite how it reads
   in a source list.

## Teammate 4: QA and demo operations

Start now:
1. The closed-loop comparison (`closed_loop.py`, or the matching section in
   the app) is the moment to build the demo around: show the Digital Shadow
   number breach, then show the Digital Twin number hold steady, same
   starting point. That is a stronger live moment than the single forecast
   chart.
2. Write the runbook for the live demo: which 3 to 4 presets get clicked,
   in what order, and what each should show on screen.
3. Block time on Day 2 to record the backup video once the app is
   stable. Do this before the final hour, not during it.
4. Test the hybrid-weight slider at both extremes (0 and 1) yourself once
   the app is running, and write down what should happen at each so you can
   tell if something is broken versus just an extreme but valid value.

## For the technical lead

Run, in order, once: `data_sanity_check.py`, `src/features.py`,
`src/rule_model.py`, `src/simulator.py`, `src/recommender.py`,
`src/ml_model.py`, then `streamlit run src/app.py`. Everything above already
ran successfully against the real downloaded data before this runbook was
written.
