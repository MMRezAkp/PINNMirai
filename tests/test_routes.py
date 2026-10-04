import pandas as pd
import pytest

from mirai_pinn.routes import validate_route, validate_route_split


def test_validate_route_adds_zero_grade_and_derives_positive_dt():
    """A route missing grade gets a usable zero-grade timing contract."""
    route = validate_route(pd.DataFrame({"time_s": [0.0, 1.0], "speed_mps": [0.0, 2.0]}))
    assert route["road_grade_deg"].tolist() == [0.0, 0.0]
    assert route["dt_s"].tolist() == [1.0, 1.0]


def test_validate_route_rejects_non_increasing_time():
    """A repeated timestamp cannot define an acceleration or episode order."""
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_route(pd.DataFrame({"time_s": [0.0, 0.0], "speed_mps": [0.0, 1.0]}))


def test_route_split_rejects_overlapping_names():
    """An overlap would leak a training route into the generalization report."""
    with pytest.raises(ValueError, match="overlap"):
        validate_route_split(["UDDS"], ["UDDS"])
