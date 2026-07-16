"""Reduced-order battery physics used as soft constraints, not OEM calibration."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class BatteryParameters:
    capacity_ah: float = 23.51  # 1.6 kWh * 1000 Wh/kWh / 245 V
    nominal_voltage_v: float = 245.0
    soc_min: float = 0.0
    soc_max: float = 1.0

def coulomb_count_step(soc: torch.Tensor, current_a: torch.Tensor, dt_s: torch.Tensor,
                       capacity_ah: float, charge_positive: bool = False) -> torch.Tensor:
    """One-step SOC propagation. Default convention: positive current discharges."""
    sign = 1.0 if charge_positive else -1.0
    return (soc + sign * current_a * dt_s / (capacity_ah * 3600.0)).clamp(0.0, 1.0)

def ocv_from_soc(soc: torch.Tensor) -> torch.Tensor:
    """Polynomial copied from the supplied Mirai surrogate; calibrate before use."""
    return 1722.4*soc**5 - 4747.5*soc**4 + 4891.9*soc**3 - 2312.1*soc**2 + 525.72*soc + 200.06
