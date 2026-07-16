from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mirai_pinn.data import load_target_excel

parser = argparse.ArgumentParser(description="Standardise CAN/Hioki battery telemetry.")
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
frame = load_target_excel(args.input)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(args.output, index=False)
print(f"Wrote {len(frame):,} rows to {args.output}")
