from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save_prediction_figure(actual, predicted, out_path, title="Prediction vs Actual"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(actual, cmap="coolwarm")
    plt.title("Actual")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(predicted, cmap="coolwarm")
    plt.title("Predicted")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_uncertainty_figure(std_map, out_path, title="Uncertainty Map"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.imshow(std_map, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
