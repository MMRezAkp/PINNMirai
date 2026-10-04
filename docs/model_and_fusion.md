# Model, transfer learning and fusion

The data branch is an LSTM with feature order `[voltage_V, current_A, temperature_K]`. This matches the supplied implementation's three-feature sequence convention, but the default hidden size is 32 rather than its 10. A source checkpoint therefore needs matching dimensions or conversion before loading.

The physics branch applies Coulomb counting at every step. Repository convention: **positive current means discharge**. Verify this against target telemetry; a reversed sign invalidates the physical loss.

The fusion gate receives the three measurements plus data and physics SOC candidates and returns a convex weight. A weight near one trusts the learned estimate; near zero trusts physics propagation. `fused_loss` has four terms: SOC supervision, one-step Coulomb consistency, low-weight OCV voltage consistency and a small gate regulariser.

Recommended experiments: source LSTM; target-only LSTM; frozen transfer then fine-tuned transfer; PINN-only; full fusion. Report RMSE, MAE, maximum error, SOC-bound violations, and metrics by temperature/SOC bins.

## Controller boundary

`MiraiEnergyEnvironment` turns the physical surrogate state into the exact
estimator measurement sequence `[voltage_v, current_a, temperature_k]`, then
passes the fused SOC and gate to the controller. Its normal policy observation
is `[traction_power_kw, fused_soc, fusion_gate, battery_temperature_k,
previous_fc_power_kw, battery_soh]`. `true_soc` is diagnostic information,
not a normal controller feature. `oracle_soc=True` is allowed only for a
paired experimental baseline that quantifies the value lost to estimator
error; it must not be presented as deployable controller performance.
