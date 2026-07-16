"""Losses for the fused SOC estimator."""
from __future__ import annotations
import torch
from torch import nn
from .physics import BatteryParameters, coulomb_count_step, ocv_from_soc

def fused_loss(output: dict[str, torch.Tensor], features: torch.Tensor, dt_s: torch.Tensor,
               target_soc: torch.Tensor, battery: BatteryParameters = BatteryParameters()) -> dict[str, torch.Tensor]:
    prediction = output["soc"]
    label = nn.functional.mse_loss(prediction, target_soc)
    expected_next = coulomb_count_step(prediction[:, :-1], features[:, 1:, 1], dt_s[:, 1:], battery.capacity_ah)
    physics = nn.functional.mse_loss(prediction[:, 1:], expected_next)
    voltage = nn.functional.mse_loss(ocv_from_soc(prediction), features[:, :, 0])
    gate_regularizer = (output["gate"] * (1 - output["gate"])).mean()
    return {"total": label + 0.5 * physics + 0.01 * voltage + 1e-3 * gate_regularizer,
            "label": label, "physics": physics, "voltage": voltage}
