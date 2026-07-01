# AirTwin: Task List for Final Build Day

Wednesday, July 1, 2026. This is a same-day ship. Tiers: P0 breaks the demo
if skipped, P1 meaningfully strengthens it, P2 cut first without debate.
Work top to bottom inside each tier. Claude Code and Codex should run in
parallel on separate scope, not sequentially; Codex does not need to wait
for Claude Code, the model artifacts it needs already exist in models/.

Coordination rule: Claude Code owns src/app.py, src/closed_loop.py, and any
new UI files. Codex owns a new src/evaluation.py (or an evaluation/
directory) and should not edit Claude Code's files, to avoid merge
conflicts from two agents in the same repo at once.

## Before any code: human, five minutes, blocking

- [ ] Confirm the Digital Twin versus Digital Shadow requirement wording
      against the actual course lecture slide or kickoff material (see
      Session_update.md section 5, item 1). If it requires literal physical
      actuation, stop and reassess scope before continuing below.

## P0, Claude Code

- [x] Verify environment: `uv sync`, then `uv run streamlit run src/app.py`,
      confirm it boots with no errors.
- [x] Room visual: replace the placeholder metric cards with an actual room
      panel (SVG or styled HTML), occupancy icons, a fan and window icon
      that change state, and a green, yellow, red overlay tied to
      rule_result["status"].
- [x] Confirm presets A, B, D still produce sane numbers after any changes
      above; do not let UI work silently change simulator constants.
- [x] Confirm the closed-loop comparison section (Digital Twin versus
      Digital Shadow) still renders and the 40-versus-0 result still holds
      after UI changes.
- [x] Add an evaluation panel to the dashboard displaying whatever
      src/evaluation.py (Codex, below) produces; if that is not ready yet,
      stub it with the numbers already in Session_update.md section 4 so
      the panel is not empty during rehearsal.
- [x] Pass over app.py for rough edges: loading state on first run, a
      visible label when no trained model is found, no raw stack traces
      reaching the screen.

## P0, Codex

- [x] Build src/evaluation.py: score rules-only, ML-only, and hybrid risk
      against the same held-out chronological split, each reporting F1,
      precision, recall, and a confusion matrix. This is the ablation the
      original plan called for and it does not exist yet.
- [x] Compute lead time before breach: for every true positive, how many
      minutes of warning the model gave versus the actual crossing.
- [x] Formalize the closed-loop comparison (forced do_nothing versus
      automatic control) into a reusable function that returns minutes
      above threshold avoided and peak CO2 for both runs, so the dashboard
      and the report can call the same function instead of two different
      hand-written numbers.
- [x] Review app.py and closed_loop.py for edge cases: zero occupants,
      hybrid weight at exactly 0 or exactly 1, the first few minutes of a
      run before there is enough history for the ML features.
- [x] Produce the final numbers and a short written summary for the
      report's evaluation section.

## P0, human and team

- [ ] Confirm requirement wording (see top of this file, blocking).
- [ ] Write and rehearse the demo script: opening line, preset click order,
      and the exact sentence to say at the closed-loop moment ("same
      starting point, same physics, the only difference is whether the
      system is allowed to act on its own").
- [ ] Record the backup video once the app stops changing. Do this before
      the last hour, not during it.
- [ ] Push the repo to GitHub. It already has a first commit and is ready
      to push as-is.
- [ ] Draft the written report. The decision log in Session_update.md
      section 3 can be reused almost verbatim for the methodology section.
- [ ] Build the slide deck: problem, architecture, the closed-loop proof
      with the 40-versus-0 numbers, evaluation honesty (high recall, low
      precision, and why), known limits, close.

## P1

- [ ] Preset C: an outdoor-air-quality scenario where filtration is
      preferred over opening a window. This needs a plausible synthetic
      outdoor PM value since OpenAQ was cut; do not wire in a live feed.
- [ ] UI polish pass: consistent spacing and color, remove the default
      Streamlit look.
- [ ] Turn the README's "known dataset limits to say out loud if asked"
      section into an actual slide, not just a note to remember live.

## P2, cut first if time is short

- [ ] Physical hardware element (a smart plug or relay switching a small
      fan from closed_loop.py's action_taken). Only attempt if a part is
      already on hand; do not go buy or source anything today.
- [ ] Additional classifiers or hyperparameter tuning beyond the two
      already trained.
- [ ] Room visual animation beyond static state changes.
- [ ] Any live OpenAQ or other external API integration.

## Definition of done for today

- [x] App runs locally with at least three demo presets.
- [x] Closed-loop comparison is visible and correct in the dashboard.
- [x] Evaluation numbers (rules, ML, hybrid) exist somewhere the team can
      point to, dashboard or report, not only in a terminal.
- [ ] Repo pushed to GitHub.
- [ ] Backup video recorded.
- [ ] Demo script rehearsed at least once, out loud, against a clock.
- [ ] Written report and slide deck exist in at least a complete draft.

## Starter prompts

Paste this into a new Claude Code session:

> Read Session_update.md and task.md in full before changing anything.
> Confirm you understand the decision log and the verified numbers. Start
> with the first unchecked task under "P0, Claude Code." Do not touch
> anything listed under Codex's scope. After finishing a task, check it off
> in task.md and log any new decision in Session_update.md before starting
> the next one.

Paste this into a new Codex session:

> Read Session_update.md and task.md in full before changing anything.
> Your scope is only the tasks under "P0, Codex." Work in a new
> src/evaluation.py so you do not conflict with parallel work on app.py.
> Use the existing data and model artifacts, do not retrain unless a task
> explicitly asks for it. After finishing a task, check it off in task.md.
