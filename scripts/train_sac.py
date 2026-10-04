"""Train a SAC controller on named route CSVs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mirai_pinn.environment import MiraiEnergyEnvironment
from mirai_pinn.routes import load_route, validate_route_split
from mirai_pinn.sac import SACAgent, SACConfig
parser = argparse.ArgumentParser(description="Train SAC on disjoint named driving routes.")
parser.add_argument("--train", nargs="+", required=True, metavar="NAME=CSV"); parser.add_argument("--evaluation", nargs="*", default=[], metavar="NAME=CSV")
parser.add_argument("--episodes", type=int, default=100); parser.add_argument("--checkpoint", default="artifacts/sac_mirai.pt")
args = parser.parse_args()
def parse(items): return {item.split("=", 1)[0]: load_route(item.split("=", 1)[1]) for item in items}
train, evaluation = parse(args.train), parse(args.evaluation); validate_route_split(train.keys(), evaluation.keys())
agent = SACAgent(6, 1, SACConfig())
for episode in range(args.episodes):
    env = MiraiEnergyEnvironment(train[list(train)[episode % len(train)]]); observation, _ = env.reset(); done = False
    while not done:
        action = agent.act(observation); next_observation, reward, done, _, _ = env.step(action)
        agent.observe(observation, action, reward, next_observation, done); agent.update(); observation = next_observation
agent.save(args.checkpoint); print(f"Saved SAC checkpoint: {args.checkpoint}")
