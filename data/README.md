# Data layout

Place unmodified source files in `data/raw/` and prepared CSV files in `data/processed/`; both paths are ignored by Git. The supplied workbook remains outside the repository because telemetry provenance and redistribution rights need confirmation.
# Controller route files

SAC route files are CSVs with `time_s` and `speed_mps`; `road_grade_deg` is
optional and defaults to zero. Times must be strictly increasing, speeds must
be finite and non-negative, and each route needs at least two rows. Name train
and evaluation routes separately: an evaluation name may not also be a training
name, because that would invalidate the held-out generalization report.
