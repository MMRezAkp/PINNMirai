"""Driving-route contracts used for reproducible controller experiments."""
from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ROUTE_COLUMNS = ("time_s", "speed_mps")


def validate_route(route: pd.DataFrame, name: str = "route") -> pd.DataFrame:
    """Return a finite, time-ordered route with grade and timestep columns."""
    missing = set(REQUIRED_ROUTE_COLUMNS) - set(route.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if len(route) < 2:
        raise ValueError(f"{name} must contain at least two rows")

    out = route.copy()
    if "road_grade_deg" not in out:
        out["road_grade_deg"] = 0.0
    out = out.loc[:, ["time_s", "speed_mps", "road_grade_deg"]].reset_index(drop=True)
    values = out.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite time, speed, and grade values")
    if (out["speed_mps"] < 0).any():
        raise ValueError(f"{name} speed_mps must be non-negative")
    intervals = out["time_s"].diff().iloc[1:]
    if (intervals <= 0).any():
        raise ValueError(f"{name} time_s must be strictly increasing")
    out["dt_s"] = np.r_[float(intervals.iloc[0]), intervals.to_numpy(dtype=float)]
    return out


def load_route(path: str | Path) -> pd.DataFrame:
    """Load and validate a route CSV, using its filename in validation errors."""
    route_path = Path(path)
    return validate_route(pd.read_csv(route_path), name=route_path.name)


def validate_route_split(train_names: Collection[str], evaluation_names: Collection[str]) -> None:
    """Reject route-name overlap that would invalidate held-out evaluation."""
    overlap = set(train_names) & set(evaluation_names)
    if overlap:
        raise ValueError(f"Training/evaluation route overlap: {sorted(overlap)}")
