# Mirai PINN Energy Management

[![tests](https://github.com/OWNER/mirai-pinn-energy-management/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/mirai-pinn-energy-management/actions/workflows/tests.yml)

A reproducible research scaffold for Toyota Mirai-style fuel-cell hybrid energy management. It combines a reduced-order battery model, a physics-informed neural network (PINN), transfer learning from a source-cell SOC model, and a learned fusion layer. It is not OEM calibration or a Toyota-affiliated product.

## What the supplied files contribute

| Layer | Role | Supplied source |
| --- | --- | --- |
| Vehicle/EMS | UDDS traction demand, fuel-cell/battery split, hydrogen and ageing cost | `SAC_DNN_Mirai_UDDS.ipynb` |
| Source estimator | LSTM using voltage, current, temperature to estimate SOC | `archive (11) (1)/lstm_soc_prediction.py` |
| Target data adapter | Reads CAN/Hioki-style Excel columns | `combined_data3.xlsx` |
| Fused estimator | PINN SOC plus transfer LSTM plus confidence gate | `src/mirai_pinn/models.py` |

The notebook's battery is a **1.6 kWh pack surrogate** (`Q_bat=1.6`) with nominal charge calculated using a 245 V reference. The source LSTM documentation describes Tesla Model 3 2170/NCA cell data. Those are different electrochemical systems, so this project uses Tesla only as a pre-training source and requires target-domain fine-tuning/calibration before EMS use.

## Method

`[V, I, T] -> transfer LSTM -> SOC_data`

`[I, dt, SOC_0] -> PINN -> SOC_physics`, constrained by `dSOC/dt = -I/Q`

`SOC = gate * SOC_data + (1 - gate) * SOC_physics`

The gate is bounded in `[0, 1]`. Training combines labelled SOC error, Coulomb-counting residual, voltage consistency and gate regularisation. A calibrated equivalent-circuit or electro-thermal model can replace the default surrogate.

## Quick start

```powershell
cd mirai-pinn-energy-management
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts\prepare_excel.py --input "..\archive (11) (1)\combined_data3.xlsx" --output data\processed\target.csv
pytest -q
python scripts\train_fused.py --data data\processed\target.csv --epochs 50 --source-checkpoint path\to\source_lstm.pth
```

The training entry point is intentionally usable without a source checkpoint; it explicitly reports **target-only, not transfer learning** in that condition. For a quick smoke run, append `--max-rows 64`.

## Layout

```text
src/mirai_pinn/       reusable data, physics, model and training code
scripts/              preparation and training entry points
tests/                unit and interface contracts
docs/                 model, battery/data and validation documentation
configs/              explicit experiment settings
data/                 ignored telemetry; tracked schema only
```

Read [the model/fusion notes](docs/model_and_fusion.md), [battery/data statement](docs/battery_and_data.md), and [testing strategy](docs/testing.md) before interpreting results.

## Safety and limitations

This is research code. It does not command a vehicle, replace an automotive BMS, or validate safety limits. Validate against held-out target-vehicle cycles and enforce independent SOC, current, voltage and temperature bounds before hardware-in-the-loop work.
