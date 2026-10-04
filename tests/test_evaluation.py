import numpy as np
import pandas as pd

from mirai_pinn.evaluation import evaluate_agent


class ZeroAgent:
    def act(self, observation, deterministic=False):
        return np.zeros(1, dtype=np.float32)


def test_evaluation_reports_route_and_aggregate_metrics():
    """A held-out route report must expose controller, health, and estimator outcomes."""
    route = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "speed_mps": [0.0, 5.0, 6.0]})
    report = evaluate_agent(ZeroAgent(), {"unseen": route})
    expected = {"route", "cumulative_reward", "hydrogen_g", "terminal_soc", "soc_violations",
                "power_violations", "temperature_violations", "battery_soh_change",
                "fuel_cell_soh_change", "fused_soc_mae"}
    assert expected.issubset(report.columns)
    assert set(report["route"]) == {"unseen", "aggregate"}
