from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mirai_pinn.models import FusedSOCEstimator
from mirai_pinn.training import fused_loss

parser = argparse.ArgumentParser(description="Train the fused SOC estimator on prepared telemetry.")
parser.add_argument("--data", required=True); parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--checkpoint", default="artifacts/fused_soc.pt")
parser.add_argument("--source-checkpoint", help="Optional compatible LSTM pre-training checkpoint.")
parser.add_argument("--max-rows", type=int, help="Optional contiguous debugging/training window.")
args = parser.parse_args()
df = pd.read_csv(args.data)
if args.max_rows: df = df.iloc[:args.max_rows].copy()
required = ["voltage_v", "current_a", "temperature_k", "dt_s", "soc"]
if set(required) - set(df.columns): raise ValueError(f"CSV must include {required}")
x = torch.tensor(df[["voltage_v", "current_a", "temperature_k"]].to_numpy(), dtype=torch.float32).unsqueeze(0)
dt = torch.tensor(df["dt_s"].to_numpy(), dtype=torch.float32).unsqueeze(0)
y = torch.tensor(df["soc"].to_numpy(), dtype=torch.float32).unsqueeze(0)
model = FusedSOCEstimator()
if args.source_checkpoint:
    model.load_source_checkpoint(args.source_checkpoint, freeze_encoder=True)
    print(f"Loaded source checkpoint and froze encoder: {args.source_checkpoint}")
else:
    print("No source checkpoint supplied: training is target-only, not transfer learning.")
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
for epoch in range(args.epochs):
    opt.zero_grad(); result = fused_loss(model(x, dt, y[:, 0]), x, dt, y); result["total"].backward(); opt.step()
    print(f"epoch={epoch + 1} total={result['total'].item():.6f} label={result['label'].item():.6f}")
Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), args.checkpoint)
