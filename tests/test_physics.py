import torch
from mirai_pinn.physics import coulomb_count_step

def test_positive_discharge_current_lowers_soc():
    actual = coulomb_count_step(torch.tensor([0.6]), torch.tensor([10.0]), torch.tensor([3600.0]), 20.0)
    assert torch.allclose(actual, torch.tensor([0.1]))
