import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Linear(64 * 8 * 8, hidden_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)

class CNNLSTM(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=128, forecast_steps=6, out_channels=5, h=32, w=64):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, forecast_steps * out_channels * h * w)
        self.forecast_steps = forecast_steps
        self.out_channels = out_channels
        self.h = h
        self.w = w

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = []
        for i in range(t):
            feats.append(self.encoder(x[:, i]))
        feats = torch.stack(feats, dim=1)
        out, _ = self.lstm(feats)
        last = out[:, -1]
        pred = self.head(last)
        pred = pred.view(b, self.forecast_steps, self.out_channels, self.h, self.w)
        return pred
