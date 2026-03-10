import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "predict", "full"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="transformer")
    args = parser.parse_args()

    script_map = {
        "train": "src/train.py",
        "evaluate": "src/evaluate.py",
        "predict": "src/inference.py",
        "full": "src/run_full.py",
    }

    cmd = [sys.executable, script_map[args.command], "--config", args.config]
    if args.command in {"train", "evaluate", "predict"}:
        cmd.extend(["--model", args.model])
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
