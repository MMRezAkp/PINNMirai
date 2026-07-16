# Model, transfer learning and fusion

The data branch is an LSTM with feature order `[voltage_V, current_A, temperature_K]`. This matches the supplied implementation's three-feature sequence convention, but the default hidden size is 32 rather than its 10. A source checkpoint therefore needs matching dimensions or conversion before loading.

The physics branch applies Coulomb counting at every step. Repository convention: **positive current means discharge**. Verify this against target telemetry; a reversed sign invalidates the physical loss.

The fusion gate receives the three measurements plus data and physics SOC candidates and returns a convex weight. A weight near one trusts the learned estimate; near zero trusts physics propagation. `fused_loss` has four terms: SOC supervision, one-step Coulomb consistency, low-weight OCV voltage consistency and a small gate regulariser.

Recommended experiments: source LSTM; target-only LSTM; frozen transfer then fine-tuned transfer; PINN-only; full fusion. Report RMSE, MAE, maximum error, SOC-bound violations, and metrics by temperature/SOC bins.
