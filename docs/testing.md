# Test and validation strategy

| Test | What it protects |
| --- | --- |
| `test_data.py` | telemetry mapping, SOC percentage conversion and derived timestep |
| `test_physics.py` | current sign and Coulomb-counting update |
| `test_fusion.py` | tensor interface and bounded SOC/gate outputs |

These unit tests are not evidence of vehicle accuracy. For an experiment, use chronological cycle-level held-out splits to avoid near-duplicate sequence leakage; fit scaling only on the train split; include an ambient-temperature or drive-cycle out-of-distribution test; and compare with Coulomb counting, the original notebook state, and raw CAN SOC only when it is trusted as reference.
