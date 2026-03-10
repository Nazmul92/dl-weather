import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def load_era5_to_array(
    path_pattern: str,
    variables: list[str],
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    level: int | None = None,
) -> np.ndarray:
    ds = xr.open_mfdataset(path_pattern, combine="by_coords")
    arrays = []
    for var in variables:
        if var not in ds:
            ds.close()
            raise KeyError(f"Variable '{var}' not found in dataset.")
        da = ds[var]
        if level is not None and "level" in da.dims:
            da = da.sel(level=level)
        da = da.transpose(time_dim, lat_dim, lon_dim)
        arrays.append(da.values)
    out = np.stack(arrays, axis=1).astype(np.float32)
    ds.close()
    return out


def fill_nan_per_channel(data: np.ndarray) -> np.ndarray:
    if not np.isnan(data).any():
        return data
    means = np.nanmean(data, axis=0, keepdims=True)
    return np.where(np.isnan(data), means, data).astype(np.float32)


def normalize_per_channel(data: np.ndarray) -> tuple[np.ndarray, dict]:
    out = np.zeros_like(data, dtype=np.float32)
    stats = {}
    for c in range(data.shape[1]):
        mean = float(data[:, c].mean())
        std = float(data[:, c].std() + 1e-6)
        out[:, c] = (data[:, c] - mean) / std
        stats[str(c)] = {"mean": mean, "std": std}
    return out, stats


def main():
    parser = argparse.ArgumentParser(description="Convert ERA5 NetCDF to project NPZ format.")
    parser.add_argument("--input", required=True, help="NetCDF file, glob pattern, or directory/*.nc")
    parser.add_argument("--output", default="data/processed/era5_processed.npz")
    parser.add_argument("--stats-out", default="data/processed/era5_stats.json")
    parser.add_argument("--variables", nargs="+", required=True, help="Variables to include, e.g. t2m sp u10 v10 tp")
    parser.add_argument("--time-dim", default="time")
    parser.add_argument("--lat-dim", default="latitude")
    parser.add_argument("--lon-dim", default="longitude")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", help="Apply global per-channel normalization")
    args = parser.parse_args()

    data = load_era5_to_array(
        path_pattern=args.input,
        variables=args.variables,
        time_dim=args.time_dim,
        lat_dim=args.lat_dim,
        lon_dim=args.lon_dim,
        level=args.level,
    )
    data = fill_nan_per_channel(data)

    stats = {
        "variables": args.variables,
        "shape": list(data.shape),
        "normalized": False,
    }
    if args.normalize:
        data, norm_stats = normalize_per_channel(data)
        stats["normalized"] = True
        stats["channel_stats"] = norm_stats

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=data)

    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved tensor: {out_path} with shape {tuple(data.shape)}")
    print(f"Saved metadata: {stats_path}")


if __name__ == "__main__":
    main()
