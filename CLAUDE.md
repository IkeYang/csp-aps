# Notes for working on this repo

- Python package `cspaps` in `src/`, tests in `tests/`. Run `pytest -q` before committing.
- Coordinates: x = East, y = North, z = Up (m), origin at the tower base. Azimuths are compass (N = 0, E = 90, clockwise). Keep this convention in every module.
- All user inputs live in a case YAML (`cases/*.yaml`) parsed by `config.py`. New inputs = a new dataclass field with a default; unknown keys must keep raising errors.
- Power chain order is fixed in `simulate.STAGES`; add new loss mechanisms there rather than ad hoc.
- Physics changes need a test against an analytic result or a conservation check, not just a smoke test.
- Efficiencies come from `optics.intercept_fraction` (grid-independent); flux maps are only for distribution / peak flux.
- README is in Chinese (the user's language); code, comments and docstrings in English.
