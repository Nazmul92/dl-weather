import torch
import torch.nn as nn

class PersistenceModel(nn.Module):
    def __init__(self, forecast_steps=6):
        super().__init__()
        self.forecast_steps = forecast_steps

    def forward(self, x):
        # x: [B, T, C, H, W]
        last_frame = x[:, -1:].clone()
        return last_frame.repeat(1, self.forecast_steps, 1, 1, 1)
