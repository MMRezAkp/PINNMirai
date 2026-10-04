import numpy as np

from mirai_pinn.sac import SACAgent, SACConfig


def test_deterministic_sac_action_is_bounded_and_repeatable():
    """Evaluation cannot change a Gaussian draw between identical observations."""
    agent = SACAgent(observation_dim=6, action_dim=1, config=SACConfig(hidden_dim=16))
    first = agent.act(np.zeros(6), deterministic=True)
    assert np.allclose(first, agent.act(np.zeros(6), deterministic=True))
    assert np.all(np.abs(first) <= 1.0)


def test_sac_update_returns_losses_after_replay_warmup():
    """A full replay minibatch must drive a finite SAC update, not just storage."""
    agent = SACAgent(observation_dim=6, action_dim=1, config=SACConfig(batch_size=2, hidden_dim=16))
    for _ in range(2):
        agent.observe(np.zeros(6), np.zeros(1), 1.0, np.ones(6), False)
    losses = agent.update()
    assert {"actor_loss", "critic_loss", "alpha"}.issubset(losses)
    assert all(np.isfinite(value) for value in losses.values())


def test_public_modules_import():
    """The documented controller modules must be importable by users."""
    from mirai_pinn.environment import MiraiEnergyEnvironment
    from mirai_pinn.sac import SACAgent as ImportedAgent

    assert MiraiEnergyEnvironment.observation_dim == 6
    assert ImportedAgent is SACAgent
