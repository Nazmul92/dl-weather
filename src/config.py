from pathlib import Path
import yaml

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_output_dirs(cfg):
    out = Path(cfg.get("output_dir", "outputs"))
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out
