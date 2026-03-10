import argparse
import json
from pathlib import Path
import numpy as np
import torch

from config import load_config
from data_loader import build_dataloaders
from utils import set_seed, get_device, load_checkpoint
from train import build_model
from losses.metrics import compute_metrics
from visualization.maps import save_prediction_figure

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="transformer", choices=["persistence", "lstm", "cnn_lstm", "transformer"])
    parser.add_argument("--metrics-out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu"))

    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(args.model, cfg).to(device)

    ckpt = Path(cfg.get("output_dir", "outputs")) / "checkpoints" / f"{args.model}_best.pt"
    if args.model != "persistence" and ckpt.exists():
        model = load_checkpoint(model, str(ckpt), device)
        print(f"Loaded checkpoint: {ckpt}")

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = compute_metrics(trues, preds)
    print("Test metrics:", metrics)
    print("METRICS_JSON:", json.dumps(metrics))

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Save one example figure for channel 0, first forecast step
    actual = trues[0, 0, 0]
    predicted = preds[0, 0, 0]
    fig_path = Path(cfg.get("output_dir", "outputs")) / "figures" / f"{args.model}_prediction.png"
    save_prediction_figure(actual, predicted, fig_path, title=f"{args.model} prediction")
    print(f"Saved prediction figure to {fig_path}")

if __name__ == "__main__":
    main()
