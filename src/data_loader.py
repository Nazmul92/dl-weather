import numpy as np
import torch
import xarray as xr
from glob import glob
from torch.utils.data import Dataset, DataLoader


class WeatherSequenceDataset(Dataset):
    def __init__(self, data, input_steps=12, forecast_steps=6):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps

    def __len__(self):
        return len(self.data) - self.input_steps - self.forecast_steps + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_steps]
        y = self.data[idx + self.input_steps: idx + self.input_steps + self.forecast_steps]
        return x, y

def load_npz_data(npz_path):
    arr = np.load(npz_path)["data"].astype(np.float32)
    return arr


def load_era5_data(cfg):
    era5 = cfg["dataset"]["era5"]
    data_path = era5["path"]
    variables = era5["variables"]
    time_dim = era5.get("time_dim", "time")
    lat_dim = era5.get("lat_dim", "latitude")
    lon_dim = era5.get("lon_dim", "longitude")
    level = era5.get("level", None)

    matched = sorted(glob(data_path))
    if len(matched) == 0:
        raise FileNotFoundError(f"No NetCDF files matched: {data_path}")
    if len(matched) == 1:
        ds = xr.open_dataset(matched[0])
    else:
        # Avoid xarray+dask requirement from open_mfdataset in minimal environments.
        datasets = [xr.open_dataset(p) for p in matched]
        ds = xr.combine_by_coords(datasets)
    arrays = []
    for var in variables:
        if var not in ds:
            ds.close()
            raise KeyError(f"Variable '{var}' not found in ERA5 dataset.")
        da = ds[var]
        if level is not None and "level" in da.dims:
            da = da.sel(level=level)
        da = da.transpose(time_dim, lat_dim, lon_dim)
        arrays.append(da.values)

    data = np.stack(arrays, axis=1).astype(np.float32)
    ds.close()

    if np.isnan(data).any():
        fill_values = np.nanmean(data, axis=0, keepdims=True)
        data = np.where(np.isnan(data), fill_values, data)
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid split ratios for dataset length. Adjust train_ratio/val_ratio.")
    return data[:train_end], data[train_end:val_end], data[val_end:]


def normalize_by_train(train_data, val_data, test_data):
    mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
    std = train_data.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std
    return train_norm, val_norm, test_norm


def build_dataloaders(cfg):
    kind = cfg["dataset"].get("kind", "npz")
    if kind == "npz":
        npz_path = cfg["dataset"]["npz_path"]
        data = load_npz_data(npz_path)
    elif kind == "era5_nc":
        data = load_era5_data(cfg)
    else:
        raise ValueError(f"Unsupported dataset.kind='{kind}'. Use 'npz' or 'era5_nc'.")

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=cfg["dataset"]["train_ratio"],
        val_ratio=cfg["dataset"]["val_ratio"],
    )
    train_data, val_data, test_data = normalize_by_train(train_data, val_data, test_data)

    input_steps = cfg["data"]["input_steps"]
    forecast_steps = cfg["data"]["forecast_steps"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 0)

    # Data-driven shape setup for easier ERA5 usage on custom subsets.
    channels, height, width = train_data.shape[1], train_data.shape[2], train_data.shape[3]
    model_cfg = cfg["model"]
    if model_cfg.get("auto_shape", True):
        model_cfg["channels"] = int(channels)
        model_cfg["height"] = int(height)
        model_cfg["width"] = int(width)
    elif (
        model_cfg["channels"] != channels
        or model_cfg["height"] != height
        or model_cfg["width"] != width
    ):
        raise ValueError(
            "Model input shape does not match dataset shape. "
            f"Config model=[C:{model_cfg['channels']},H:{model_cfg['height']},W:{model_cfg['width']}], "
            f"data=[C:{channels},H:{height},W:{width}]."
        )

    train_ds = WeatherSequenceDataset(train_data, input_steps, forecast_steps)
    val_ds = WeatherSequenceDataset(val_data, input_steps, forecast_steps)
    test_ds = WeatherSequenceDataset(test_data, input_steps, forecast_steps)

    if len(train_ds) <= 0 or len(val_ds) <= 0 or len(test_ds) <= 0:
        raise ValueError(
            "One of train/val/test sequence datasets is empty. "
            "Increase time length or reduce input_steps/forecast_steps."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
