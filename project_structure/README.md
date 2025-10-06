# Project Structure Scaffold

This directory gives you a clean layout to move the existing scripts into when you are ready.

## Suggested layout

- `src/core/` – shared models, configuration, and reusable helpers.
- `src/animations/` – animation logic such as `leaf_animation.py`.
- `src/strategies/` – different rake or bagging strategies.
- `scripts/` – entry points or CLI utilities you run directly.
- `data/` – raw or processed datasets used by the simulations.
- `notebooks/` – exploratory analysis or experiments in notebook form.
- `tests/` – automated tests for the refactored code.

You can create additional folders as needed while migrating the code. Existing files in the project root were not modified.

## Newly extracted modules

The original `with_bagging.py` logic was lifted into reusable pieces under `src/leaf_raking/`:

- `core/calibration.py`, `core/distribution.py`, `core/piles.py`, `core/timeline.py`, `core/bagging.py` – build the calibrated yard state, outside-in schedules, and bagging plans.
- `strategies/front_sweep.py` – front-sweep pass timing and column-aware spillage helpers.
- `animations/with_bagging.py` – packaged version of the legacy animation that imports the shared pieces.

`with_bagging.py` in the project root remains unchanged for reference, so you can diff behaviour as you continue migrating.
