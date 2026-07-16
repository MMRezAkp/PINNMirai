import torch
from mirai_pinn.models import FusedSOCEstimator

def test_fused_model_returns_bounded_soc_and_gate():
    output = FusedSOCEstimator(hidden=8)(torch.randn(2, 5, 3), torch.full((2, 5), 0.1), torch.tensor([0.6, 0.6]))
    assert output["soc"].shape == (2, 5)
    assert torch.all((output["soc"] >= 0) & (output["soc"] <= 1))
    assert torch.all((output["gate"] >= 0) & (output["gate"] <= 1))
