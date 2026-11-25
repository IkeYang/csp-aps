# Aksai field test data

This directory collects the source observations, prepared inputs, and test outputs for the Aksai heliostat field.

## Source data

- `定日镜、云预测相机坐标汇总表.xlsx` contains the heliostat coordinates and four camera positions.
- `SW_MAIN.csv` and `NE_SLAVE.csv` contain the two DNI measurements used as spatial anchors. SW is the main station and NE is the slave station.
- `NW_SLAVE.csv` and `SE_SLAVE.csv` are retained as supplied. They do not contain DNI columns.
- The station records cover 2026-06-18 at approximately 20-second intervals.

## Prepared model inputs

- `layout_full.csv` contains all 11,960 heliostats, with their original IDs and local ENU coordinates.
- `camera_positions.csv` contains the SW and NE camera positions transformed from the plant building coordinates into the field ENU frame. The affine fit to heliostat coordinates has sub-millimetre residuals.
- `case.yaml` is the CSP-APS single-instant test case. It places the receiver bottom and top at 127 m and 177 m, with a 9.05 m outside radius and 16 circumferential panels.
- `nasa_power_dni_2026.csv` contains the NASA POWER background for a short date window around the field measurements. It is not a full-year weather file and should not be used for annual-yield estimates.
- `dni_map_2026-06-18_120008.csv` and its JSON record contain the inferred spatial DNI map used for the single-instant test.

The test map linearly interpolates the SW and NE measurements along the station axis. Perpendicular to that axis, it fades toward the NASA POWER `ALLSKY_SFC_SW_DNI` background with a 600 m Gaussian scale. This is a convenient spatial test scenario, not a measured ground DNI map. The selected shared sample is 2026-06-18 12:00:08 local time: SW 864.990 W/m2, NE 988.238 W/m2, and NASA POWER background 857.382 W/m2.

The site uses 94.396 E, 38.9132 N, and 2,902 m elevation. The spreadsheet ground elevations imply a local datum about 2,904.919 m above the listed geodetic elevation. The prepared mirror-axis z coordinate adds a 5 m test pivot height above each local ground elevation.

The supplied information does not specify heliostat dimensions, pivot height, reflectivity, tower diameter, or allowable receiver flux. For this test case those values use the software demo defaults: 8 x 8 m mirrors, 5 m pivot height, 0.94 reflectivity, 8 m tower diameter, and 1,000 kW/m2 flux limit. The receiver is represented by the software's cylindrical surface with 16 panels. The model does not explicitly represent planar facets of a regular 16-sided prism.

## Prepare another DNI map

Run from the repository root. The script reads the original station CSVs and the prepared camera and layout files, and downloads a short NASA POWER date window when needed.

```bash
HTTPS_PROXY=http://127.0.0.1:7897 python scripts/adapt_aksai_test.py \
  --input cases/aksai_field_test_2026 \
  --output cases/aksai_field_test_2026 \
  --time "2026-06-18 12:00:00"
```

## Run the test

Use the prepared DNI map for a full-field centre-aim snapshot:

```bash
cspaps snapshot cases/aksai_field_test_2026/case.yaml \
  --time "2026-06-18 12:00:08" \
  --dni-map cases/aksai_field_test_2026/dni_map_2026-06-18_120008.csv \
  --gpus all \
  --out results/aksai_field_test_2026
```

Run fast APS aiming with the same inputs:

```bash
cspaps aps cases/aksai_field_test_2026/case.yaml \
  --time "2026-06-18 12:00:08" \
  --dni-map cases/aksai_field_test_2026/dni_map_2026-06-18_120008.csv \
  --method greedy --mode fast --no-certify --gpus all \
  --out results/aksai_field_test_2026
```

The saved test results are under `test_results/`. Centre aiming exceeds the assumed 1,000 kW/m2 limit. Greedy APS aiming passes the flux check, with 11,960 mirrors active, 392.083 MW absorbed power, and a verified 906.51 kW/m2 peak in the CPU run. The second saved APS result has the same reported power and peak. Its exported result does not record a GPU device ID, so the result alone cannot verify which device ran it. Exact run timing depends on hardware and GPU count.
