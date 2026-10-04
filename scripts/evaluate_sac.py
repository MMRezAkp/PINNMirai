"""Evaluate a saved SAC controller without learning on held-out route CSVs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mirai_pinn.evaluation import evaluate_agent
from mirai_pinn.routes import load_route
from mirai_pinn.sac import SACAgent
parser = argparse.ArgumentParser(description="Evaluate SAC checkpoint on held-out routes.")
parser.add_argument("--checkpoint", required=True); parser.add_argument("--routes", nargs="+", required=True, metavar="NAME=CSV")
parser.add_argument("--output", default="artifacts/sac_evaluation.csv"); parser.add_argument("--oracle-soc", action="store_true")
args = parser.parse_args()
routes = {item.split("=", 1)[0]: load_route(item.split("=", 1)[1]) for item in args.routes}
report = evaluate_agent(SACAgent.load(args.checkpoint), routes, oracle_soc=args.oracle_soc)
Path(args.output).parent.mkdir(parents=True, exist_ok=True); report.to_csv(args.output, index=False); print(report.to_string(index=False))
