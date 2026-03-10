import argparse
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

from config import load_config, ensure_output_dirs
from data_loader import build_dataloaders
from utils import set_seed, get_device, save_checkpoint
from models.persistence import PersistenceModel
from models.lstm_model import LSTMForecast
from models.cnn_lstm import CNNLSTM
from models.transformer_model import WeatherTransformer

def build_model(model_name, cfg):
    model_cfg = cfg["model"]
    input_steps = cfg["data"]["input_steps"]
    forecast_steps = cfg["data"]["forecast_steps"]

    if model_name == "persistence":
        return PersistenceModel(forecast_steps=forecast_steps)
    if model_name == "lstm":
        return LSTMForecast(
            channels=model_cfg["channels"],
            height=model_cfg["height"],
            width=model_cfg["width"],
            hidden_dim=model_cfg["hidden_dim"],
            forecast_steps=forecast_steps,
        )
    if model_name == "cnn_lstm":
        return CNNLSTM(
            in_channels=model_cfg["channels"],
            hidden_dim=model_cfg["hidden_dim"],
            forecast_steps=forecast_steps,
            out_channels=model_cfg["channels"],
            h=model_cfg["height"],
            w=model_cfg["width"],
        )
    if model_name == "transformer":
        return WeatherTransformer(
            in_channels=model_cfg["channels"],
            forecast_steps=forecast_steps,
            img_h=model_cfg["height"],
            img_w=model_cfg["width"],
            input_steps=input_steps,
            patch_size=model_cfg["patch_size"],
            emb_dim=model_cfg["emb_dim"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"Unknown model: {model_name}")

def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total += loss.item()
    return total / max(len(loader), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="transformer", choices=["persistence", "lstm", "cnn_lstm", "transformer"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_output_dirs(cfg)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu"))

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(args.model, cfg).to(device)

    criterion = nn.MSELoss()

    if args.model == "persistence":
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Persistence model - validation loss: {val_loss:.6f}")
        return

    train_cfg = cfg["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )

    best_val = float("inf")
    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            loop.set_postfix(train_loss=f"{loss.item():.4f}")

        train_loss = total / max(len(train_loader), 1)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir / "checkpoints" / f"{args.model}_best.pt"
            save_checkpoint(model, str(ckpt))
            print(f"Saved best checkpoint to {ckpt}")

if __name__ == "__main__":
    main()
