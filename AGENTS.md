# Repository Guidelines

## Project Structure & Module Organization

AirTwin is a Python/Streamlit project for room-level CO2 forecasting and ventilation recommendations. Core implementation lives in `src/`: `data_loader.py` loads the UCI occupancy files, `features.py` builds time-series features, `rule_model.py` and `ml_model.py` score risk, `simulator.py` and `recommender.py` evaluate actions, `closed_loop.py` demonstrates automatic control, and `app.py` is the dashboard. Raw datasets are in `data/raw/`; trained artifacts are written to `models/`. Supporting notes and team process live in `docs/`, `README.md`, `CLAUDE.md`, and `Session_update.md`.

## Build, Test, and Development Commands

Use `uv` for dependency management and command execution.

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python data_sanity_check.py`: verify dataset shape, timestamps, and class balance.
- `uv run python src/features.py`: sanity-check feature and label generation.
- `uv run python src/rule_model.py`: run the deterministic rule baseline.
- `uv run python src/simulator.py`: inspect intervention curves.
- `uv run python src/recommender.py`: test action selection.
- `uv run python src/ml_model.py`: train classifiers and save `.joblib` files to `models/`.
- `uv run streamlit run src/app.py`: launch the interactive dashboard.

Run the scripts in this order on a fresh checkout so model artifacts exist before using the full dashboard.

## Coding Style & Naming Conventions

Target Python 3.12. Follow the existing style: 4-space indentation, descriptive `snake_case` functions and variables, uppercase module constants such as `MODEL_PATH`, and small functions with typed public signatures where practical. Keep imports grouped as standard library, third-party, then local modules. Prefer pandas/scikit-learn APIs over manual parsing for tabular data. Preserve the time-series invariant in `features.py`: features look backward only, labels look forward only.

## Testing Guidelines

There is no dedicated test suite yet. Treat the command sequence above as the current smoke test, especially `data_sanity_check.py`, `src/features.py`, `src/ml_model.py`, and `src/closed_loop.py`. When adding tests, place them under `tests/`, name files `test_*.py`, and focus on leakage prevention, chronological splits, recommender behavior, and Streamlit-safe model loading.

## Commit & Pull Request Guidelines

Git history is not available from this working directory, so no project-specific commit convention can be inferred. Use concise imperative commits, for example `Add closed-loop comparison metrics`. Pull requests should describe the behavioral change, list commands run, mention any regenerated model artifacts, and include screenshots for dashboard changes.

## Security & Configuration Tips

Do not commit downloaded archives, credentials, or local environment files. Keep raw source data under `data/raw/` and model outputs under `models/`. If recalibrating simulator constants or demo presets, document the rationale in the PR because those values affect the story shown in the dashboard.
