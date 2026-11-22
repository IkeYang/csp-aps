# Notes for working on this repo

- Python package `cspaps` in `src/`, tests in `tests/`. Install with `pip install -e ".[dev]"`; run `pytest -q` (~8 min) before committing.
- Coordinates: x = East, y = North, z = Up (m), origin at the tower base. Azimuths are compass (N = 0, E = 90, clockwise). Keep this convention in every module.
- All user inputs live in a case YAML (`cases/*.yaml`) parsed by `config.py`. New inputs = a new dataclass field with a default (the GUI form picks it up automatically; add a Chinese label in `gui_app.LABELS`). Unknown keys must keep raising errors.
- Layers: `optics`/`shading`/`sun`/`receiver` (physics, panel limits) -> `simulate.evaluate` (one sun state, power chain `STAGES`) -> `aps` (ResponseSet + AimProblem, solvers, verification loop), `uniformity`, `certify` (upper bounds / optimality gaps for APS and layout, relaxation rounding), `engine` (online cycle), `realtime` (replay), `annual`, `layout_opt` -> `mcrt`/`soltrace` (validation only) -> `workflows` (run folders, files, plots) -> `cli` and `gui_app`.
- Spatial DNI and the annual layout value: `dni_maps` (site rasters, bilinear, `box_max` for bounds) -> `scenarios` (scenario sets: sun, hours, map per scenario; regime-wise binning preserves DNI energy; save / load / fingerprint) -> `layout_value` (`solve_instant` = task 1 with fixed inner settings on a layout-independent grid; `LayoutValue`, V = E - lambda_L Pi, status rules, exports) -> `layout_opt.formal_ranking`. Samples may carry maps (`Samples.dni_at`); `simulate.annual` then gives every mirror its own DNI.
- The formal value must stay reproducible: same layout + scenario set + inner settings -> bit-identical P_s, U_s (no wall-clock budgets inside `solve_instant`). Compare layouts only on the same scenario fingerprint and settings digest.
- Flux limits: always compare modelled incident flux with `receiver.incident_limit_w_m2` (panel limits, height profile, absorbed basis, bias). Feasibility statements must come from `verify.verify_beams` (fine grid, full beams), never from the coarse map alone.
- Efficiencies come from `optics.intercept_fraction` (grid-independent); flux maps are for distribution / peak flux.
- Replay time base: sample-and-hold intervals (`realtime.replay_times`); never count samples x step.
- Layout CSVs carry heliostat IDs and micrometre precision; site checks use `field.site_mask` (1 mm tolerance) everywhere and report dropped heliostats.
- Physics changes need a test against an analytic result, a conservation check, brute force or MCRT, not just a smoke test. The advisor's acceptance cases live in `tests/test_p0_regressions.py` and friends.
- GUI tests run with `CSPAPS_GUI_SYNC=1` (background jobs run synchronously).
- Optimality statements: a bound must be valid for the candidate model (any plan / layout the model allows); test new bounds against brute force or SCIP (`tests/test_certify.py`). Never call a heuristic result "optimal" without a certificate.
- Docs: `docs/MODEL.md` (model), `docs/REVIEW_RESPONSE.md` (advisor review items -> implementation). README and docs are in Chinese (the user's language); code, comments and docstrings in English.
