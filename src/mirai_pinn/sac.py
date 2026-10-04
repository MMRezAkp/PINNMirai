"""A compact, standard Soft Actor-Critic implementation for the Mirai environment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SACConfig:
    hidden_dim: int = 256
    replay_capacity: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.01
    learning_rate: float = 5e-4
    target_entropy: float | None = None
    seed: int = 42
    device: str = "cpu"


class ReplayBuffer:
    """Fixed-capacity, array-backed transition storage."""

    def __init__(self, observation_dim: int, action_dim: int, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float,
            next_observation: np.ndarray, done: bool) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Replay buffer has fewer transitions than batch_size")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return tuple(torch.as_tensor(values[indices], device=device) for values in (
            self.observations, self.actions, self.rewards, self.next_observations, self.dones
        ))

    def __len__(self) -> int:
        return self.size


class _Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(observation)
        return self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent = distribution.rsample()
        action = torch.tanh(latent)
        log_probability = (distribution.log_prob(latent) - torch.log(1.0 - action.square() + 1e-6)).sum(dim=-1, keepdim=True)
        return action, log_probability

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(observation)[0])


class _Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((observation, action), dim=-1))


class SACAgent:
    """Twin-critic SAC agent with automatic entropy-temperature tuning."""

    def __init__(self, observation_dim: int, action_dim: int, config: SACConfig = SACConfig()):
        self.observation_dim, self.action_dim, self.config = observation_dim, action_dim, config
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.actor = _Actor(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic1 = _Critic(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic2 = _Critic(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic1 = _Critic(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic2 = _Critic(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(action_dim)
        self.replay = ReplayBuffer(observation_dim, action_dim, config.replay_capacity, config.seed)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor.deterministic(tensor) if deterministic else self.actor.sample(tensor)[0]
        return action.cpu().numpy()[0].astype(np.float32)

    def observe(self, observation: np.ndarray, action: np.ndarray, reward: float,
                next_observation: np.ndarray, done: bool) -> None:
        self.replay.add(np.asarray(observation, dtype=np.float32), np.asarray(action, dtype=np.float32),
                        float(reward), np.asarray(next_observation, dtype=np.float32), bool(done))

    def update(self) -> dict[str, float] | None:
        if len(self.replay) < self.config.batch_size:
            return None
        observations, actions, rewards, next_observations, dones = self.replay.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            next_actions, next_log_probabilities = self.actor.sample(next_observations)
            next_q = torch.minimum(self.target_critic1(next_observations, next_actions), self.target_critic2(next_observations, next_actions))
            target_q = rewards + self.config.gamma * (1.0 - dones) * (next_q - self.alpha.detach() * next_log_probabilities)
        critic1_loss = F.mse_loss(self.critic1(observations, actions), target_q)
        critic2_loss = F.mse_loss(self.critic2(observations, actions), target_q)
        self.critic1_optimizer.zero_grad(); critic1_loss.backward(); self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(); critic2_loss.backward(); self.critic2_optimizer.step()
        policy_actions, log_probabilities = self.actor.sample(observations)
        q_policy = torch.minimum(self.critic1(observations, policy_actions), self.critic2(observations, policy_actions))
        actor_loss = (self.alpha.detach() * log_probabilities - q_policy).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha * (log_probabilities + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
        with torch.no_grad():
            for source, target in ((self.critic1, self.target_critic1), (self.critic2, self.target_critic2)):
                for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                    target_parameter.lerp_(source_parameter, self.config.tau)
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float((critic1_loss + critic2_loss).item()), "alpha": float(self.alpha.item())}

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"observation_dim": self.observation_dim, "action_dim": self.action_dim, "config": asdict(self.config),
                    "actor": self.actor.state_dict(), "critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict(),
                    "target_critic1": self.target_critic1.state_dict(), "target_critic2": self.target_critic2.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu(), "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic1_optimizer": self.critic1_optimizer.state_dict(), "critic2_optimizer": self.critic2_optimizer.state_dict(),
                    "alpha_optimizer": self.alpha_optimizer.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "SACAgent":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = SACConfig(**checkpoint["config"], device=map_location)
        agent = cls(checkpoint["observation_dim"], checkpoint["action_dim"], config)
        for name in ("actor", "critic1", "critic2", "target_critic1", "target_critic2"):
            getattr(agent, name).load_state_dict(checkpoint[name])
        agent.log_alpha.data.copy_(checkpoint["log_alpha"].to(agent.device))
        for name in ("actor_optimizer", "critic1_optimizer", "critic2_optimizer", "alpha_optimizer"):
            getattr(agent, name).load_state_dict(checkpoint[name])
        return agent
