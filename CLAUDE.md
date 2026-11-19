# Notes for working on this repo

- Python package `cspaps` in `src/`, tests in `tests/`. Install with `pip install -e ".[dev]"`; run `pytest -q` (~1 min) before committing.
- Coordinates: x = East, y = North, z = Up (m), origin at the tower base. Azimuths are compass (N = 0, E = 90, clockwise). Keep this convention in every module.
- All user inputs live in a case YAML (`cases/*.yaml`) parsed by `config.py`. New inputs = a new dataclass field with a default (the GUI form picks it up automatically; add a Chinese label in `gui_app.LABELS`). Unknown keys must keep raising errors.
- Layers: `optics`/`shading`/`sun` (physics) -> `simulate.evaluate` (one sun state, power chain `STAGES`) -> `aps` (inner loop), `realtime`, `annual`, `layout_opt` (outer loop), `mcrt` (validation only) -> `workflows` (files + plots) -> `cli` and `gui_app`.
- Efficiencies come from `optics.intercept_fraction` (grid-independent); flux maps are for distribution / peak flux and the APS constraint.
- Physics changes need a test against an analytic result, a conservation check or MCRT, not just a smoke test.
- Model documentation: `docs/MODEL.md` (Chinese). README and docs are in Chinese (the user's language); code, comments and docstrings in English.
