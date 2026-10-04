"""Notebook-faithful reduced-order Mirai SAC environment.

This module is a research surrogate.  Its equations and coefficients follow
the supplied UDDS SAC notebook; they are not Toyota/OEM calibration.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import FusedSOCEstimator
from .physics import ocv_from_soc, safe_quadratic_current
from .routes import validate_route


class MiraiEnergyEnvironment:
    """Constrained fuel-cell power-split environment driven by a route dataframe.

    Public power values are kW.  The notebook battery-current equations use W
    internally.  Action `-1` maps to zero fuel-cell power and `+1` maps to
    `max_fc_power_kw`; all feasibility projection is performed here.
    """

    observation_dim = 6

    def __init__(
        self, route: pd.DataFrame, estimator: nn.Module | None = None, *,
        oracle_soc: bool = False, max_fc_power_kw: float = 112.0,
        max_battery_power_kw: float = 22.4, initial_soc: float = 0.6,
        initial_temperature_k: float = 298.15,
    ):
        self.route = validate_route(route)
        self.estimator = estimator if estimator is not None else FusedSOCEstimator()
        self.estimator.eval()
        self.oracle_soc = oracle_soc
        self.max_fc_power_kw = float(max_fc_power_kw)
        self.max_battery_power_kw = float(max_battery_power_kw)
        self.initial_soc = float(initial_soc)
        self.initial_temperature_k = float(initial_temperature_k)
        if self.max_fc_power_kw <= 0 or self.max_battery_power_kw <= 0:
            raise ValueError("Maximum fuel-cell and battery powers must be positive")

        # Vehicle constants and maps retained from SAC_DNN_Mirai_UDDS.ipynb.
        self.q_battery_kwh = 1.6
        self.q_nom_c = self.q_battery_kwh * 1000.0 * 3600.0 / 245.0
        self.mass_kg = 1793.45 + self.max_fc_power_kw / 1.6 + self.q_battery_kwh * 1000.0 / 24.5
        self.g, self.cd, self.area_m2, self.rho = 9.81, 0.29, 2.23, 1.2
        self.wheel_radius_m, self.rolling, self.gear_ratio, self.gear_efficiency = 0.316, 0.01, 9.09, 0.98
        self.wheel_inertia, self.em_inertia, self.auxiliary_w = 0.32, 0.008, 440.0
        self._em_speed_rpm = np.array([106.383, 510.638, 1127.66, 1744.68, 2404.26, 2936.17, 3234.04,
                                       3404.26, 3617.02, 3936.17, 4297.87, 4747.47, 5553.19, 6638.3,
                                       7553.19, 8127.66, 9063.83, 9978.72, 10851.1, 11595.7, 12106.4,
                                       12787.2, 13127.7])
        self._em_torque_nm = np.array([335, 335, 335, 335, 335, 335, 335, 335, 335, 319.687, 297.253,
                                       274.826, 252.401, 235.957, 195.616, 161.268, 143.371, 132.934,
                                       119.528, 109.114, 100.194, 92.7616, 88.3066])
        omega = self._em_speed_rpm * 2.0 * np.pi / 60.0
        self._max_em_power_kw = self._em_torque_nm * omega / 1000.0
        self._speed_for_power = omega / self.gear_ratio * self.wheel_radius_m
        self._fc_power_ratio = np.array([0.000711683, 0.057973684, 0.154842105, 0.291286842, 0.447229825,
                                         0.571645614, 0.702773684, 0.831429825, 0.875987719])
        self._eta_stack = np.array([0.61072, 0.634147, 0.618024, 0.563347, 0.519156, 0.491697, 0.446014,
                                    0.395806, 0.372961])
        self._eta_system = np.array([0.61072, 0.634147, 0.618024, 0.563347, 0.519156, 0.491697, 0.446014,
                                     0.395806, 0.372961])
        self._c_rates = np.array([0.5, 2.0, 6.0, 10.0])
        self._aging_m = np.array([31630.0, 21681.0, 12934.0, 15512.0])
        self.reset()

    def _traction_power_kw(self, index: int) -> float:
        row = self.route.iloc[index]
        velocity = float(row.speed_mps)
        previous = float(self.route.iloc[max(0, index - 1)].speed_mps)
        acceleration = (velocity - previous) / float(row.dt_s)
        grade = math.radians(float(row.road_grade_deg))
        rolling_force = self.mass_kg * self.g * self.rolling * math.cos(grade) if velocity > 0 else 0.0
        force = rolling_force + self.mass_kg * self.g * math.sin(grade) + 0.5 * self.rho * self.cd * self.area_m2 * velocity**2 + self.mass_kg * acceleration
        wheel_omega = velocity / self.wheel_radius_m
        em_omega = wheel_omega * self.gear_ratio
        wheel_torque = 4.0 * self.wheel_inertia * acceleration / self.wheel_radius_m + force * self.wheel_radius_m
        gearbox_torque = wheel_torque / (self.gear_efficiency * self.gear_ratio) if wheel_torque >= 0 else wheel_torque * self.gear_efficiency / self.gear_ratio
        torque = np.clip(self.em_inertia * acceleration / self.wheel_radius_m * self.gear_ratio + gearbox_torque,
                         -np.interp(abs(em_omega) * 60.0 / (2.0 * np.pi), self._em_speed_rpm, self._em_torque_nm),
                         np.interp(abs(em_omega) * 60.0 / (2.0 * np.pi), self._em_speed_rpm, self._em_torque_nm))
        mechanical_power_w = torque * em_omega
        em_efficiency = 0.90  # The notebook efficiency map is bounded around this operating-point surrogate.
        electrical_kw = ((mechanical_power_w / em_efficiency if torque >= 0 else mechanical_power_w * em_efficiency) + self.auxiliary_w) / 1000.0
        return float(min(electrical_kw, np.interp(velocity, self._speed_for_power, self._max_em_power_kw)))

    @staticmethod
    def _ocv(soc: float) -> float:
        return float(ocv_from_soc(torch.tensor(soc)).item())

    @staticmethod
    def _resistance(soc: float, charging: bool) -> float:
        if charging:
            return 0.0056*soc**4 - 0.0254*soc**3 + 0.0372*soc**2 - 0.0203*soc + 0.0223
        return 0.0188*soc**4 - 0.0547*soc**3 + 0.0765*soc**2 - 0.0457*soc + 0.0349

    def _project_power(self, requested_fc_kw: float, demand_kw: float) -> tuple[float, float, float, bool]:
        fc = float(np.clip(requested_fc_kw, 0.0, self.max_fc_power_kw))
        battery = demand_kw - fc
        clipped = abs(battery) > self.max_battery_power_kw
        battery = float(np.clip(battery, -self.max_battery_power_kw, self.max_battery_power_kw))
        delivered = fc + battery
        return fc, battery, float(demand_kw - delivered), clipped

    def _estimate_soc(self, current_a: float) -> tuple[float, float]:
        voltage = self._ocv(self.soc)
        features = torch.tensor([[[voltage, current_a, self.temperature_k]]], dtype=torch.float32)
        dt = torch.tensor([[max(self.current_dt_s, 1e-3)]], dtype=torch.float32)
        initial_soc = torch.tensor([self.soc], dtype=torch.float32)
        with torch.no_grad():
            output = self.estimator(features, dt, initial_soc)
        if not isinstance(output, dict) or "soc" not in output:
            raise ValueError("Estimator must return a dictionary containing 'soc'")
        soc = float(torch.as_tensor(output["soc"]).reshape(-1)[-1].item())
        gate = float(torch.as_tensor(output.get("gate", 1.0)).reshape(-1)[-1].item())
        return float(np.clip(soc, 0.0, 1.0)), float(np.clip(gate, 0.0, 1.0))

    def _observation(self, demand_kw: float, current_a: float) -> np.ndarray:
        fused_soc, gate = self._estimate_soc(current_a)
        visible_soc = self.soc if self.oracle_soc else fused_soc
        return np.array([demand_kw, visible_soc, gate, self.temperature_k, self.previous_fc_kw, self.battery_soh], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
        if seed is not None:
            np.random.seed(seed)
        self.index = 0
        self.soc = float(np.clip(self.initial_soc, 0.0, 1.0))
        self.temperature_k = self.initial_temperature_k
        self.battery_soh = 1.0
        self.fuel_cell_soh = 1.0
        self.previous_fc_kw = 0.0
        self.current_dt_s = float(self.route.iloc[0].dt_s)
        demand = self._traction_power_kw(0)
        return self._observation(demand, 0.0), {"true_soc": self.soc, "route_index": 0.0}

    def step(self, action: float | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = float(np.asarray(action).reshape(-1)[0])
        bounded_action = float(np.clip(raw_action, -1.0, 1.0))
        demand = self._traction_power_kw(self.index)
        requested_fc = self.max_fc_power_kw * (bounded_action + 1.0) / 2.0
        fc_kw, battery_kw, unserved_kw, battery_clipped = self._project_power(requested_fc, demand)

        dt_s = float(self.route.iloc[self.index].dt_s)
        self.current_dt_s = dt_s
        ocv = self._ocv(self.soc)
        discharge = battery_kw >= 0.0
        resistance = self._resistance(self.soc, charging=not discharge)
        current_a, discriminant_clipped = safe_quadratic_current(ocv, resistance, abs(battery_kw) * 1000.0, charging=not discharge)
        delta_soc = -(current_a * dt_s / self.q_nom_c) if discharge else -(current_a * dt_s / self.q_nom_c)
        self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

        c_rate = abs(current_a) / (self.q_nom_c / 3600.0)
        m = float(np.interp(c_rate, self._c_rates, self._aging_m))
        q_loss = max(m * math.exp((-31700.0 + 370.3 * c_rate) / (8.31 * self.temperature_k)), 1e-12)
        ah_total = (20.0 / q_loss) ** 0.55
        delta_battery_soh = -abs(current_a) * dt_s / (ah_total * 3600.0 * 2.0) * 0.01
        self.battery_soh = max(0.0, self.battery_soh + delta_battery_soh)
        power_ratio = fc_kw / self.max_fc_power_kw
        eta_stack = float(np.interp(power_ratio, self._fc_power_ratio, self._eta_stack))
        eta_system = float(np.interp(power_ratio, self._fc_power_ratio, self._eta_system))
        hydrogen_g_s = 0.0 if fc_kw <= 0 else 8.5e-5 * ((eta_stack / eta_system * fc_kw) ** 2) + 9.1e-3 * (eta_stack / eta_system * fc_kw) + 0.0064
        high_load = dt_s / 3600.0 * 1e-5 if power_ratio >= 0.8 else 0.0
        low_load = dt_s / 3600.0 * 8.662e-6 if 0.0 < power_ratio <= 0.2 else 0.0
        load_change = 4.185e-8 * abs(fc_kw - self.previous_fc_kw) / self.max_fc_power_kw if self.index else 0.0
        start_stop = 13.79e-8 if fc_kw != 0.0 and self.previous_fc_kw == 0.0 else 0.0
        fc_degradation = high_load + low_load + load_change + start_stop
        self.fuel_cell_soh = max(0.0, self.fuel_cell_soh - fc_degradation / 0.07)
        thermal_resistance = max(resistance, 1e-4)
        self.temperature_k += ((298.15 - self.temperature_k) / (1.052 * 791.19) + thermal_resistance * current_a**2 / 50.07) * dt_s
        self.temperature_k = float(np.clip(self.temperature_k, 250.0, 400.0))
        soc_cost = 1.7e-5 * self.max_fc_power_kw**2.9 * self.q_battery_kwh**2 * (0.575 - self.soc)**2
        real_cost = 32.94e-3 * hydrogen_g_s + self.q_battery_kwh * 139.0 * abs(delta_battery_soh) + self.max_fc_power_kw * 93.0 * (fc_degradation / 0.07)
        reward = -(real_cost + soc_cost + max(unserved_kw, 0.0))
        self.previous_fc_kw = fc_kw
        self.index += 1
        terminated = self.index >= len(self.route) - 1
        next_demand = 0.0 if terminated else self._traction_power_kw(self.index)
        observation = self._observation(next_demand, current_a)
        info: dict[str, Any] = {
            "true_soc": self.soc, "traction_power_kw": fc_kw + battery_kw, "requested_traction_power_kw": demand,
            "fuel_cell_power_kw": fc_kw, "battery_power_kw": battery_kw, "unserved_power_kw": unserved_kw,
            "action_clipped": raw_action != bounded_action, "battery_power_clipped": battery_clipped,
            "battery_discriminant_clipped": discriminant_clipped, "battery_temperature_k": self.temperature_k,
            "battery_soh": self.battery_soh, "fuel_cell_soh": self.fuel_cell_soh, "hydrogen_g_s": hydrogen_g_s,
            "real_cost": real_cost, "fused_soc": float(observation[1] if not self.oracle_soc else self._estimate_soc(current_a)[0]),
            "fusion_gate": float(observation[2]), "soc_violation": self.soc in (0.0, 1.0),
            "power_violation": bool(battery_clipped or abs(unserved_kw) > 1e-9),
            "temperature_violation": not (273.15 <= self.temperature_k <= 333.15),
        }
        return observation, float(reward), terminated, False, info
