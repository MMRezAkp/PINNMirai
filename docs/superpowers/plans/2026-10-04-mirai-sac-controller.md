# Mirai SAC Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable SAC energy-management pipeline that uses the fused PINN SOC estimate and evaluates on disjoint driving routes.

**Architecture:** The notebook's Mirai equations are moved into a deterministic, unit-documented environment.  The environment is the only component allowed to project an SAC power command to feasible fuel-cell/battery powers; it obtains the policy-visible SOC from the existing fused estimator.  A self-contained PyTorch SAC implementation trains on named routes and evaluates a checkpoint without learning.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pandas, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-mirai-sac-controller-design.md`

## Global Constraints

- Preserve the supplied `SAC_DNN_Mirai_UDDS.ipynb` equations as the reduced-order surrogate reference; do not claim OEM calibration.
- Positive battery current means discharge; power is kW externally and W within the preserved battery equations where required.
- Normal policy observations use fused SOC and never true SOC; `oracle_soc=True` is diagnostic-only.
- The action is exactly one normalized fuel-cell command in `[-1, 1]`; feasibility projection belongs to the environment.
- Evaluation route identities must be disjoint from training route identities and evaluation must not update the agent.
- Keep all transitions finite; expose clipping/infeasibility via info metrics instead of hiding it.
- This extracted directory has no `.git`; omit commit commands and report that limitation.

## Review Focus

- Routes with repeated/decreasing timestamps must fail validation before a training episode starts (Task 1).
- Battery-power clipping must recompute actual fuel-cell power so the split remains algebraically consistent (Task 2).
- The quadratic-current discriminant can become negative on infeasible power and must remain finite while recording a violation (Task 2).
- A non-oracle observation must remain unchanged when only internal true SOC is perturbed (Task 3).
- Evaluation must reject overlapping train/evaluation identities before checkpoint loading or metric reporting (Task 5).

---

### Task 1: Route data contracts and split validation

**Files:**
- Create: `src/mirai_pinn/routes.py`
- Create: `tests/test_routes.py`
- Modify: `configs/experiment.yaml`
- Modify: `data/README.md`

**Interfaces:**
- Produces: `validate_route(route: pd.DataFrame, name: str = "route") -> pd.DataFrame`, `load_route(path: str | Path) -> pd.DataFrame`, and `validate_route_split(train_names: Collection[str], evaluation_names: Collection[str]) -> None`.
- Consumes: CSV columns `time_s`, `speed_mps`, and optional `road_grade_deg`.

- [ ] **Step 1: Write failing route-contract tests**

```python
def test_validate_route_adds_zero_grade_and_derives_positive_dt():
    route = validate_route(pd.DataFrame({"time_s": [0., 1.], "speed_mps": [0., 2.]}))
    assert route["road_grade_deg"].tolist() == [0.0, 0.0]
    assert route["dt_s"].tolist() == [1.0, 1.0]

def test_validate_route_rejects_non_increasing_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_route(pd.DataFrame({"time_s": [0., 0.], "speed_mps": [0., 1.]}))

def test_route_split_rejects_overlapping_names():
    with pytest.raises(ValueError, match="overlap"):
        validate_route_split(["UDDS"], ["UDDS"])
```

- [ ] **Step 2: Run route tests to verify they fail**

Run: `pytest tests/test_routes.py -v`

Expected: FAIL because `mirai_pinn.routes` does not exist.

- [ ] **Step 3: Implement route validation and document route CSVs**

Implement the three declared interfaces. Require at least two rows, finite non-negative speed, finite grade, and strictly increasing time. Derive `dt_s` from adjacent timestamps and assign the first interval from the second. Add named `routes.train` and `routes.evaluation` examples to configuration and document the CSV schema plus disjoint split rule.

- [ ] **Step 4: Run route tests to verify they pass**

Run: `pytest tests/test_routes.py -v`

Expected: PASS.

### Task 2: Notebook-faithful Mirai dynamics and constrained environment

**Files:**
- Create: `src/mirai_pinn/environment.py`
- Create: `tests/test_environment.py`
- Modify: `src/mirai_pinn/physics.py`
- Modify: `src/mirai_pinn/__init__.py`

**Interfaces:**
- Consumes: validated routes from Task 1 and `FusedSOCEstimator` from the existing model module.
- Produces: `MiraiEnergyEnvironment(route: pd.DataFrame, estimator: nn.Module | None = None, *, oracle_soc: bool = False, max_fc_power_kw: float = 112.0, max_battery_power_kw: float = 22.4)`, `reset(seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]`, and `step(action: float | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float | bool]]`.

- [ ] **Step 1: Write failing environment tests**

```python
def test_step_projects_out_of_range_action_and_keeps_power_balance(route):
    env = MiraiEnergyEnvironment(route)
    env.reset()
    _, _, _, _, info = env.step(np.array([3.0]))
    assert 0.0 <= info["fuel_cell_power_kw"] <= env.max_fc_power_kw
    assert info["battery_power_kw"] == pytest.approx(
        info["traction_power_kw"] - info["fuel_cell_power_kw"]
    )

def test_infeasible_battery_equation_returns_finite_transition(route):
    env = MiraiEnergyEnvironment(route)
    env.reset()
    _, reward, _, _, info = env.step(np.array([-1.0]))
    assert np.isfinite(reward)
    assert np.isfinite(info["true_soc"])
    assert "battery_discriminant_clipped" in info
```

- [ ] **Step 2: Run environment tests to verify they fail**

Run: `pytest tests/test_environment.py -v`

Expected: FAIL because `MiraiEnergyEnvironment` does not exist.

- [ ] **Step 3: Implement the environment and guarded notebook equations**

Port traction-power, electric-machine limits, OCV/resistance, current/SOC, SOH, hydrogen, fuel-cell degradation, cost/reward, and thermal equations from the SAC notebook into focused private methods. Preserve notebook coefficients. Add `safe_quadratic_current(...) -> tuple[float, bool]` to physics for discriminant clipping. Map a clipped action to power, project the complete power split, and report `action_clipped`, `battery_power_clipped`, `unserved_power_kw`, and `battery_discriminant_clipped` in info.

- [ ] **Step 4: Run environment tests and existing physics tests**

Run: `pytest tests/test_environment.py tests/test_physics.py -v`

Expected: PASS.

### Task 3: Explicit estimator-controller observation boundary

**Files:**
- Modify: `src/mirai_pinn/environment.py`
- Modify: `tests/test_environment.py`
- Modify: `docs/model_and_fusion.md`

**Interfaces:**
- Produces: six-value `np.ndarray` observations ordered `[traction_power_kw, fused_soc, fusion_gate, battery_temperature_k, previous_fc_power_kw, battery_soh]`; `info["true_soc"]` remains diagnostics only.

- [ ] **Step 1: Write a failing estimator-boundary test**

```python
def test_normal_observation_uses_estimator_soc_not_true_soc(route):
    estimator = FixedEstimator(soc=0.42, gate=0.75)
    env = MiraiEnergyEnvironment(route, estimator=estimator, oracle_soc=False)
    observation, _ = env.reset()
    assert observation[1] == pytest.approx(0.42)
    assert observation[2] == pytest.approx(0.75)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_environment.py::test_normal_observation_uses_estimator_soc_not_true_soc -v`

Expected: FAIL because observations do not yet call the estimator.

- [ ] **Step 3: Implement measurement adaptation and oracle isolation**

Build `[voltage_v, current_a, temperature_k]` from the environment state and run the estimator under `torch.no_grad()`. Validate the estimator output shape and clamp fused SOC/gate to `[0, 1]`. Use true SOC only when `oracle_soc=True`. Document feature order, unit/sign conventions, and this diagnostic exception.

- [ ] **Step 4: Run relevant tests**

Run: `pytest tests/test_environment.py tests/test_fusion.py -v`

Expected: PASS.

### Task 4: Standard SAC controller module

**Files:**
- Create: `src/mirai_pinn/sac.py`
- Create: `tests/test_sac.py`
- Modify: `src/mirai_pinn/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: fixed-size NumPy observations and one-dimensional normalized actions from Task 2.
- Produces: `SACConfig`, `ReplayBuffer`, and `SACAgent`; `SACAgent.act(observation: np.ndarray, deterministic: bool = False) -> np.ndarray`, `observe(...) -> None`, `update() -> dict[str, float] | None`, `save(path: str | Path) -> None`, and `load(path: str | Path, map_location: str = "cpu") -> SACAgent`.

- [ ] **Step 1: Write failing SAC tests**

```python
def test_deterministic_sac_action_is_bounded_and_repeatable():
    agent = SACAgent(observation_dim=6, action_dim=1, config=SACConfig(hidden_dim=16))
    first = agent.act(np.zeros(6), deterministic=True)
    assert np.allclose(first, agent.act(np.zeros(6), deterministic=True))
    assert np.all(np.abs(first) <= 1.0)

def test_sac_update_returns_losses_after_replay_warmup():
    agent = SACAgent(observation_dim=6, action_dim=1, config=SACConfig(batch_size=2, hidden_dim=16))
    for _ in range(2): agent.observe(np.zeros(6), np.zeros(1), 1.0, np.ones(6), False)
    losses = agent.update()
    assert {"actor_loss", "critic_loss", "alpha"}.issubset(losses)
```

- [ ] **Step 2: Run SAC tests to verify they fail**

Run: `pytest tests/test_sac.py -v`

Expected: FAIL because `mirai_pinn.sac` does not exist.

- [ ] **Step 3: Implement SAC networks, replay, update and checkpoints**

Use two Q networks and target copies; the actor emits Gaussian mean/log standard deviation, samples with reparameterization, squashes with tanh, and corrects the log probability. Implement critic, actor, entropy-temperature, and soft-target updates. Use only existing dependencies. Serialize config and all model/optimizer state required for evaluation.

- [ ] **Step 4: Run SAC tests**

Run: `pytest tests/test_sac.py -v`

Expected: PASS.

### Task 5: Training, held-out evaluation, and reporting entry points

**Files:**
- Create: `scripts/train_sac.py`
- Create: `scripts/evaluate_sac.py`
- Create: `src/mirai_pinn/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `configs/experiment.yaml`
- Modify: `README.md`
- Modify: `docs/testing.md`

**Interfaces:**
- Consumes: Task 1 split validation, Task 2 environment, Task 4 `SACAgent`, and route CSV paths.
- Produces: `evaluate_agent(agent: SACAgent, routes: Mapping[str, pd.DataFrame], *, oracle_soc: bool = False) -> pd.DataFrame` with one row per route plus an aggregate row; CLI JSON/CSV reports and controller checkpoints.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_evaluation_reports_route_and_aggregate_metrics(route):
    report = evaluate_agent(ZeroAgent(), {"unseen": route})
    assert {"route", "cumulative_reward", "hydrogen_g", "terminal_soc",
            "soc_violations", "power_violations", "temperature_violations",
            "battery_soh_change", "fuel_cell_soh_change", "fused_soc_mae"}.issubset(report.columns)
    assert set(report["route"]) == {"unseen", "aggregate"}
```

- [ ] **Step 2: Run evaluation tests to verify they fail**

Run: `pytest tests/test_evaluation.py -v`

Expected: FAIL because `evaluate_agent` does not exist.

- [ ] **Step 3: Implement pure evaluation and CLIs**

Implement evaluation with deterministic actions and no `observe`/`update` calls. Aggregate all prescribed metrics and include split identities in output. `train_sac.py` must validate no overlap before training and save a checkpoint; `evaluate_sac.py` must load it, require evaluation route names, emit metrics, and optionally produce a paired oracle report. Update config defaults and README/testing instructions with runnable commands and research limitations.

- [ ] **Step 4: Run evaluation and complete suite**

Run: `pytest -q`

Expected: PASS with the existing estimator tests and all new controller tests.

### Task 6: Package-level smoke verification and documentation consistency

**Files:**
- Modify: `README.md`
- Modify: `docs/battery_and_data.md`
- Modify: `docs/model_and_fusion.md`
- Modify: `docs/testing.md`

**Interfaces:**
- Consumes: documented CLI and data contracts from Tasks 1–5.
- Produces: a coherent, explicit reproducibility guide.

- [ ] **Step 1: Write a failing documentation smoke test**

```python
def test_public_modules_import():
    from mirai_pinn.environment import MiraiEnergyEnvironment
    from mirai_pinn.evaluation import evaluate_agent
    from mirai_pinn.sac import SACAgent
```

- [ ] **Step 2: Run it to verify package wiring**

Run: `pytest tests/test_sac.py::test_public_modules_import -v`

Expected: PASS after Tasks 2–5; if it fails, correct exports/imports only.

- [ ] **Step 3: Finalize documentation**

Describe the full estimator-to-controller flow, all state/action values and units, route split protocol, exact commands, oracle baseline restriction, surrogate limitations, and the fact that successful held-out runs evaluate—not prove—generalization.

- [ ] **Step 4: Run the full verification set**

Run: `pytest -q; python -m compileall src scripts`

Expected: all tests pass and all modules compile.
