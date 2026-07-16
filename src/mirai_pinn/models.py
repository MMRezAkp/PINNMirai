"""Transfer, physics-informed, and fused SOC estimators."""
from __future__ import annotations
import torch
from torch import nn
from .physics import BatteryParameters, coulomb_count_step

class TransferLSTM(nn.Module):
    """Three-feature source-domain sequence model."""
    def __init__(self, features: int = 3, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x)[0]).squeeze(-1)

class PhysicsSOC(nn.Module):
    def __init__(self, battery: BatteryParameters = BatteryParameters()):
        super().__init__(); self.battery = battery; self.initial_correction = nn.Parameter(torch.tensor(0.0))
    def forward(self, current_a: torch.Tensor, dt_s: torch.Tensor, initial_soc: torch.Tensor) -> torch.Tensor:
        soc = (initial_soc + self.initial_correction).clamp(0.0, 1.0); states = []
        for k in range(current_a.shape[1]):
            soc = coulomb_count_step(soc, current_a[:, k], dt_s[:, k], self.battery.capacity_ah); states.append(soc)
        return torch.stack(states, dim=1)

class FusedSOCEstimator(nn.Module):
    """Convex fusion of target-adapted data estimator and physics trajectory."""
    def __init__(self, battery: BatteryParameters = BatteryParameters(), hidden: int = 32):
        super().__init__()
        self.transfer = TransferLSTM(hidden=hidden); self.physics = PhysicsSOC(battery)
        self.gate = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, features: torch.Tensor, dt_s: torch.Tensor, initial_soc: torch.Tensor) -> dict[str, torch.Tensor]:
        data_soc = self.transfer(features); physics_soc = self.physics(features[:, :, 1], dt_s, initial_soc)
        gate = self.gate(torch.cat((features, data_soc.unsqueeze(-1), physics_soc.unsqueeze(-1)), dim=-1)).squeeze(-1)
        return {"soc": gate * data_soc + (1.0 - gate) * physics_soc, "soc_data": data_soc, "soc_physics": physics_soc, "gate": gate}
    def freeze_transfer_encoder(self) -> None:
        for parameter in self.transfer.lstm.parameters(): parameter.requires_grad = False

    def load_source_checkpoint(self, checkpoint: str, freeze_encoder: bool = True) -> None:
        """Load LSTM weights from this project or map the supplied `SOCLSTM` names.

        The original source model uses `lstm.*` and `fc.*`; this project stores
        the output layer as `head.0.*`. Dimensions must still match.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "state_dict" in state: state = state["state_dict"]
        mapped = {k.replace("fc.", "head.0."): v for k, v in state.items()
                  if k.startswith("lstm.") or k.startswith("fc.")}
        missing, unexpected = self.transfer.load_state_dict(mapped, strict=False)
        if unexpected or len(mapped) == 0:
            raise ValueError("Checkpoint does not contain compatible LSTM/fc weights.")
        if freeze_encoder: self.freeze_transfer_encoder()
