import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, channels=5, height=32, width=64, hidden_dim=128, forecast_steps=6):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        input_dim = channels * height * width
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, forecast_steps * input_dim)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b, t, -1)
        out, _ = self.lstm(x)
        last = out[:, -1]
        pred = self.head(last)
        pred = pred.view(b, self.forecast_steps, c, h, w)
        return pred
