# Mirai SAC controller integration design

## Purpose and scope

Extend the existing PINNMirai SOC-estimation scaffold into a reproducible
research pipeline for fuel-cell hybrid energy management.  The pipeline must
preserve the supplied `SAC_DNN_Mirai_UDDS.ipynb` Mirai surrogate equations as
its modelling reference while making their inputs, units, state transitions,
and constraints explicit.  The output is research code only: it is not OEM
calibration, a safety controller, or hardware-ready software.

## Existing components and source of truth

`mirai_pinn.models.FusedSOCEstimator` is the sole estimator used by the
controller in normal operation.  It consumes `[voltage_V, current_A,
temperature_K]` and returns `soc`, `soc_data`, `soc_physics`, and `gate`.
Positive battery current means discharge.

The supplied SAC notebook is the source of truth for the reduced-order Mirai
surrogate: vehicle parameters, traction-power calculation, electric-machine
limits, open-circuit voltage and resistance polynomials, battery SOH update,
thermal update, fuel-cell efficiency/hydrogen relation, fuel-cell degradation,
and cost/reward composition.  Equations are retained but moved behind named,
unit-documented Python methods.  Unsafe numerical inputs (for example a
negative square-root discriminant) produce a finite, constrained transition
and a diagnostic violation rather than NaN values.

## Architecture

### Environment

Add `mirai_pinn.environment.MiraiEnergyEnvironment`.  A route is a dataframe
with `time_s`, `speed_mps`, and optional `road_grade_deg`; acceleration is
derived from time-aware speed differences.  At each one-second-equivalent
transition the environment:

1. calculates traction/electrical demand with the notebook vehicle model;
2. receives one normalized action in `[-1, 1]` and maps it to fuel-cell power
   in `[0, P_max_fc_kw]`;
3. enforces fuel-cell and battery-power limits, retaining a record of every
   clipping or infeasible demand;
4. advances SOC, battery/fuel-cell SOH, and battery temperature using the
   notebook equations;
5. calculates hydrogen, degradation and SOC costs and returns their negative
   sum as reward;
6. prepares next estimator measurements from the surrogate state and applies
   the fused estimator to obtain the SOC presented to the controller.

The full physical state is retained for transition calculations and metrics.
It is not exposed to the learned policy as true SOC in normal operation.

### Controller interface

The SAC policy observation has six ordered, documented values:
`[traction_power_kw, fused_soc, fusion_gate, battery_temperature_K,
previous_fc_power_kw, battery_soh]`.  `fused_soc` and `fusion_gate` come from
the estimator.  `true_soc` is available only in the info dictionary and in an
explicit `oracle_soc=True` experimental mode, so estimator and controller
performance can be compared without conflation.

The action is a single normalized fuel-cell command.  The environment, not
the actor, owns all feasibility projection.  Each transition info dictionary
includes requested/actual fuel-cell and battery power, SOC and temperature,
cost components, hydrogen flow, SOH changes, estimator output, and constraint
violations.

### SAC implementation

Add `mirai_pinn.sac` with a squashed-Gaussian actor, two Q critics and target
critics, a bounded replay buffer, automatic entropy-temperature tuning, soft
target updates, and serialization.  The API separates stochastic training
actions from deterministic evaluation actions.  All random generators are
seedable.  Shapes are batch-safe and action log probabilities include the
tanh Jacobian correction.

### Training and evaluation

Add command-line scripts to train a controller on named route files and to
evaluate a saved checkpoint.  Training accepts multiple route CSVs and
samples episodes across them.  Evaluation accepts a disjoint list of named
routes/traffic patterns and never updates SAC parameters.

Reported per-route and aggregate metrics are: cumulative reward, notebook
real cost, hydrogen use, terminal SOC, SOC/power/temperature violation counts
and magnitudes, battery and fuel-cell SOH changes, and fused-SOC error versus
the environment state.  An optional paired oracle evaluation uses identical
routes and a deterministic actor with oracle SOC enabled; it is a diagnostic
baseline, not a deployment result.

## Data contracts

Route CSVs require monotonically increasing `time_s` and non-negative
`speed_mps`; `road_grade_deg` defaults to zero.  The estimator's measurement
contract remains `[voltage_v, current_a, temperature_k]`.  The environment
generates these compatible measurement values, making the estimator-controller
intersection explicit.  A configurable initial SOC/temperature/health state
is reset independently for each episode.

Route splits are named in configuration as training and unseen evaluation
routes.  No claim of generalization is made unless route identities are
disjoint and the final report lists the split.  Diverse traffic patterns may
be represented by distinct speed profiles and grade columns; their data
quality and calibration remain the experimenter's responsibility.

## Failure handling and safety boundaries

Inputs are validated at construction/reset.  Invalid routes fail early with a
clear error.  Numerical state variables are bounded to physically meaningful
surrogate limits; demand beyond feasible power is logged, not hidden.  The
agent is never allowed to bypass action projection.  Checkpoints include model
and optimizer state, dimensions, and configuration needed for evaluation.

## Tests and documentation

Tests will establish that: actions are projected to valid power; a reset/step
has finite bounded state; a controller observation uses fused rather than true
SOC; replay sampling and SAC update shapes are valid; deterministic actions
are repeatable; route splits reject overlap; and held-out evaluation reports
all requested metrics.  Existing estimator tests remain green.

README, model/fusion notes, testing guide, configuration, and the new
controller documentation will describe the full data flow, all units/sign
conventions, limitations, commands, and the difference between evaluation and
deployment claims.
