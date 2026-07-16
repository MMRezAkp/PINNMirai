# Battery and data statement

## Original Mirai notebook surrogate

The SAC notebook instantiates `Mirai_model_V2(P_max_fc=..., Q_bat=1.6)`. `Q_bat` is in kWh. Nominal charge is `(Q_bat * 1000 * 3600) / 245`, approximately **23.51 Ah** for its 1.6 kWh, 245 V surrogate. SOC is propagated from electrical power through an OCV polynomial and separate charge/discharge resistance polynomials. It also contains a two-node thermal calculation and a simple throughput-based SOH term.

This is a reduced-order research parameterisation, not Toyota's production battery specification.

## Source versus target

The supplied LSTM documentation calls its source data Tesla Model 3 4.5 Ah 2170/NCA cells with current, voltage and temperature. The workbook exposes high-voltage battery voltage/current, three temperatures and CAN SOC. Its filename and column names do not prove vehicle model, chemistry, sensor calibration, sign convention or data quality. The adapter starts with temperature sensor `tb1`; later experiments should compare mean, median and maximum temperature aggregation.

Before reporting performance, record target chemistry, series/parallel count, usable capacity, current sign, voltage scaling/offset, temperature selection, CAN SOC provenance, cycle/ambient split, and frozen/unfrozen source layers.
