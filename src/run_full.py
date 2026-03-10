import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config


MODELS = ["persistence", "lstm", "cnn_lstm", "transformer"]


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _write_summary_csv(metrics_by_model: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "MAE", "RMSE", "R2"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            row = {"model": model}
            row.update(metrics_by_model[model])
            writer.writerow(row)


def _plot_comparison(metrics_by_model: dict, out_path: Path) -> None:
    labels = MODELS
    mae_vals = [metrics_by_model[m]["MAE"] for m in labels]
    rmse_vals = [metrics_by_model[m]["RMSE"] for m in labels]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(labels, mae_vals, color="#2A9D8F")
    plt.title("MAE by Model")
    plt.ylabel("MAE")
    plt.xticks(rotation=20)

    plt.subplot(1, 2, 2)
    plt.bar(labels, rmse_vals, color="#E76F51")
    plt.title("RMSE by Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline for all models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.get("output_dir", "outputs"))
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        _run([sys.executable, "src/train.py", "--config", args.config, "--model", model])
        _run(
            [
                sys.executable,
                "src/evaluate.py",
                "--config",
                args.config,
                "--model",
                model,
                "--metrics-out",
                str(metrics_dir / f"{model}.json"),
            ]
        )

    _run([sys.executable, "src/inference.py", "--config", args.config, "--model", "transformer"])

    metrics_by_model = {}
    for model in MODELS:
        with (metrics_dir / f"{model}.json").open("r", encoding="utf-8") as f:
            metrics_by_model[model] = json.load(f)

    summary_json = metrics_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_by_model, f, indent=2)

    summary_csv = metrics_dir / "summary.csv"
    _write_summary_csv(metrics_by_model, summary_csv)

    comparison_png = output_dir / "figures" / "metrics_comparison.png"
    _plot_comparison(metrics_by_model, comparison_png)

    print(f"Saved metrics summary JSON: {summary_json}")
    print(f"Saved metrics summary CSV: {summary_csv}")
    print(f"Saved comparison figure: {comparison_png}")


if __name__ == "__main__":
    main()
