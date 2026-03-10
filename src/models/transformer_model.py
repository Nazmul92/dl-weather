import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x

class WeatherTransformer(nn.Module):
    def __init__(
        self,
        in_channels=5,
        forecast_steps=6,
        img_h=32,
        img_w=64,
        input_steps=12,
        patch_size=4,
        emb_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim)
        self.forecast_steps = forecast_steps
        self.in_channels = in_channels
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size

        num_patches = (img_h // patch_size) * (img_w // patch_size)
        total_tokens = input_steps * num_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, emb_dim))
        self.dropout = nn.Dropout(dropout)
        # Use pooled token representation to keep parameter count tractable.
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, forecast_steps * in_channels * img_h * img_w),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        tokens = []
        for i in range(t):
            tok = self.patch_embed(x[:, i])
            tokens.append(tok)
        tokens = torch.cat(tokens, dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.dropout(tokens)
        z = self.transformer(tokens)
        z = z.mean(dim=1)
        out = self.head(z)
        out = out.view(b, self.forecast_steps, self.in_channels, self.img_h, self.img_w)
        return out
