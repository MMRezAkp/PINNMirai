"""Held-out route evaluation for SAC energy-management policies."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd
from .environment import MiraiEnergyEnvironment
from .routes import validate_route

def evaluate_agent(agent, routes: Mapping[str, pd.DataFrame], *, oracle_soc: bool = False) -> pd.DataFrame:
    """Evaluate deterministic actions only; return route rows and one aggregate row."""
    rows = []
    for name, route in routes.items():
        env = MiraiEnergyEnvironment(validate_route(route, name), oracle_soc=oracle_soc)
        observation, _ = env.reset(); rewards = []; hydrogen = 0.0; errors = []
        counts = {"soc_violations": 0, "power_violations": 0, "temperature_violations": 0}; done = False
        while not done:
            observation, reward, done, _, info = env.step(agent.act(observation, deterministic=True))
            rewards.append(reward); hydrogen += float(info["hydrogen_g_s"])
            errors.append(abs(float(info["fused_soc"]) - float(info["true_soc"])))
            for key, source in (("soc_violations", "soc_violation"), ("power_violations", "power_violation"), ("temperature_violations", "temperature_violation")):
                counts[key] += int(info[source])
        rows.append({"route": name, "cumulative_reward": float(sum(rewards)), "real_cost": float(sum(-x for x in rewards)),
                     "hydrogen_g": hydrogen, "terminal_soc": float(info["true_soc"]), **counts,
                     "battery_soh_change": 1.0 - float(info["battery_soh"]), "fuel_cell_soh_change": 1.0 - float(info["fuel_cell_soh"]),
                     "fused_soc_mae": float(np.mean(errors))})
    report = pd.DataFrame(rows)
    if report.empty: raise ValueError("At least one evaluation route is required")
    aggregate = {column: ("aggregate" if column == "route" else float(report[column].mean())) for column in report.columns}
    return pd.concat([report, pd.DataFrame([aggregate])], ignore_index=True)
