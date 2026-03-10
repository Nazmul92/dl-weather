import argparse
from pathlib import Path
import torch
import numpy as np

from config import load_config
from data_loader import build_dataloaders
from utils import set_seed, get_device, load_checkpoint
from train import build_model
from visualization.maps import save_uncertainty_figure

@torch.no_grad()
def mc_dropout_predict(model, x, n_samples=10):
    model.train()  # keep dropout active
    preds = []
    for _ in range(n_samples):
        preds.append(model(x).unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    return mean_pred, std_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="transformer", choices=["persistence", "lstm", "cnn_lstm", "transformer"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu"))

    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(args.model, cfg).to(device)

    ckpt = Path(cfg.get("output_dir", "outputs")) / "checkpoints" / f"{args.model}_best.pt"
    if args.model != "persistence" and ckpt.exists():
        model = load_checkpoint(model, str(ckpt), device)

    x, y = next(iter(test_loader))
    x = x.to(device)

    if args.model == "transformer":
        mean_pred, std_pred = mc_dropout_predict(model, x, n_samples=cfg["inference"]["mc_samples"])
        std_map = std_pred[0, 0, 0].cpu().numpy()
        out_path = Path(cfg.get("output_dir", "outputs")) / "figures" / "transformer_uncertainty.png"
        save_uncertainty_figure(std_map, out_path)
        print("Mean prediction shape:", tuple(mean_pred.shape))
        print("Std prediction shape:", tuple(std_pred.shape))
        print(f"Saved uncertainty figure to {out_path}")
    else:
        with torch.no_grad():
            pred = model(x)
        print("Prediction shape:", tuple(pred.shape))

if __name__ == "__main__":
    main()
