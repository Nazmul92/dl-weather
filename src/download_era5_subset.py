import argparse
from pathlib import Path

import cdsapi


def main():
    parser = argparse.ArgumentParser(description="Download a small ERA5 subset for training.")
    parser.add_argument("--out", default="data/era5/era5_tiny_subset.nc")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--month", default="01")
    parser.add_argument("--days", type=int, default=10, help="Number of days starting from day 1.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=[
            "2m_temperature",
            "surface_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
        ],
    )
    parser.add_argument(
        "--times",
        nargs="+",
        default=["00:00", "06:00", "12:00", "18:00"],
        help="Fewer timestamps keep file smaller.",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        default=[55, -5, 45, 10],
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Small geographic box reduces memory use.",
    )
    parser.add_argument("--grid-lat", type=float, default=1.5)
    parser.add_argument("--grid-lon", type=float, default=1.5)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_list = [f"{d:02d}" for d in range(1, args.days + 1)]

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": args.variables,
            "year": args.year,
            "month": args.month,
            "day": day_list,
            "time": args.times,
            "area": args.area,  # North, West, South, East
            "grid": [args.grid_lat, args.grid_lon],
        },
        str(out_path),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
