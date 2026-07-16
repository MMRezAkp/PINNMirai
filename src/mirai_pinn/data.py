"""Data contracts and adapters; never infer battery identity from column names."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

CAN_COLUMNS = {
    "time_s": "Time__s_RawFacilities",
    "voltage_v": "HVBatt_Volt_Hioki_analog10hz__U1__V",
    "current_a": "HVBatt_Curr_Hioki_analog10hz__I1__A",
    "temperature_c": "HVBatt_temp_of_battery_tb1_EV__C",
    "soc_pct": "HVBatt_SOC_vsCAN3__per",
}

def load_target_excel(path: str | Path) -> pd.DataFrame:
    """Load and standardise the supplied CAN/Hioki-style workbook."""
    raw = pd.read_excel(path)
    missing = set(CAN_COLUMNS.values()) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required telemetry columns: {sorted(missing)}")
    out = raw.loc[:, list(CAN_COLUMNS.values())].rename(columns={v: k for k, v in CAN_COLUMNS.items()})
    out = out.dropna().sort_values("time_s").drop_duplicates("time_s").reset_index(drop=True)
    out["dt_s"] = out["time_s"].diff().fillna(0.1).clip(lower=1e-3)
    out["temperature_k"] = out["temperature_c"] + 273.15
    out["soc"] = (out["soc_pct"] / 100.0).clip(0.0, 1.0)
    return out
