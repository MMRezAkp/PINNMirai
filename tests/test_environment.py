import numpy as np
import pandas as pd
import pytest
import torch

from mirai_pinn.environment import MiraiEnergyEnvironment


class FixedEstimator(torch.nn.Module):
    def __init__(self, soc: float, gate: float):
        super().__init__()
        self.soc, self.gate = soc, gate

    def forward(self, features, dt_s, initial_soc):
        shape = features.shape[:2]
        return {
            "soc": torch.full(shape, self.soc),
            "gate": torch.full(shape, self.gate),
        }


@pytest.fixture
def route():
    return pd.DataFrame(
        {"time_s": [0.0, 1.0, 2.0], "speed_mps": [0.0, 8.0, 10.0], "road_grade_deg": [0.0, 0.0, 0.0]}
    )


def test_step_projects_out_of_range_action_and_keeps_power_balance(route):
    """A controller command above range cannot bypass the feasible power split."""
    env = MiraiEnergyEnvironment(route)
    env.reset()
    _, _, _, _, info = env.step(np.array([3.0]))
    assert 0.0 <= info["fuel_cell_power_kw"] <= env.max_fc_power_kw
    assert info["battery_power_kw"] == pytest.approx(
        info["traction_power_kw"] - info["fuel_cell_power_kw"]
    )
    assert info["action_clipped"] is True


def test_infeasible_battery_equation_returns_finite_transition(route):
    """An extreme battery demand must not turn a training transition into NaNs."""
    env = MiraiEnergyEnvironment(route, max_battery_power_kw=0.01)
    env.reset()
    _, reward, _, _, info = env.step(np.array([-1.0]))
    assert np.isfinite(reward)
    assert np.isfinite(info["true_soc"])
    assert "battery_discriminant_clipped" in info


def test_normal_observation_uses_estimator_soc_not_true_soc(route):
    """Policy input must not silently become an oracle-SOC controller."""
    env = MiraiEnergyEnvironment(route, estimator=FixedEstimator(soc=0.42, gate=0.75), oracle_soc=False)
    observation, _ = env.reset()
    assert observation[1] == pytest.approx(0.42)
    assert observation[2] == pytest.approx(0.75)
    env.soc = 0.1
    changed_state_observation = env._observation(0.0, 0.0)
    assert changed_state_observation[1] == pytest.approx(0.42)
