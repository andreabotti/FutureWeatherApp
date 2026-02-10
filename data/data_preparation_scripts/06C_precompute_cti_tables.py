#!/usr/bin/env python3
"""
06C_precompute_cti_tables.py

Reads per-station parquets written by 05C and writes app-ready tables:

Outputs under: <parquet-root>/_tables/

- D-CTI__DBT__F-DD__L-ALL.parquet
    Tidy daily stats: region, station_key, scenario, date, Tmax, Tmean, Tmin

- D-CTI__DBT__F-MM__L-ALL.parquet
    Tidy monthly stats: region, station_key, scenario, month (YYYY-MM), Tmax, Tmean, Tmin

- D-CTI__Inventory__F-NA__L-ALL.parquet
    Inventory per station: region, station_key, rows, n_cols, cols (semicolon)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DAILY_OUT_NAME = "D-CTI__DBT__F-DD__L-ALL.parquet"
MONTHLY_OUT_NAME = "D-CTI__DBT__F-MM__L-ALL.parquet"
INVENTORY_OUT_NAME = "D-CTI__Inventory__F-NA__L-ALL.parquet"


def discover_station_parquets(root: Path):
    for region_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "_tables"]):
        region = region_dir.name
        for p in sorted(region_dir.glob("*.parquet")):
            yield region, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-root", required=True, help="data/04__italy_cti_parquet")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.parquet_root).resolve()
    tables = root / "_tables"
    tables.mkdir(parents=True, exist_ok=True)

    daily_rows = []
    monthly_rows = []
    inventory_rows = []

    for region, p in discover_station_parquets(root):
        station_key = p.stem
        df = pd.read_parquet(p)

        if "dt" in df.columns:
            df = df.set_index("dt")

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

        cols = [c for c in df.columns if c is not None]
        cols_str = [str(c) for c in cols]

        inventory_rows.append(
            {
                "region": region,
                "station_key": station_key,
                "rows": int(len(df)),
                "n_cols": int(len(cols_str)),
                "cols": ";".join(cols_str),
            }
        )

        for scenario in cols_str:
            s = df[scenario].dropna()
            if s.empty:
                continue
            g = s.resample("D")
            daily = pd.DataFrame(
                {
                    "Tmax": g.max(),
                    "Tmean": g.mean(),
                    "Tmin": g.min(),
                }
            ).reset_index().rename(columns={"dt": "date"})
            daily["region"] = region
            daily["station_key"] = station_key
            daily["scenario"] = scenario
            daily_rows.append(daily)

            gm = s.resample("ME")
            monthly = pd.DataFrame(
                {
                    "Tmax": gm.max(),
                    "Tmean": gm.mean(),
                    "Tmin": gm.min(),
                }
            ).reset_index().rename(columns={"dt": "month"})
            monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
            monthly["region"] = region
            monthly["station_key"] = station_key
            monthly["scenario"] = scenario
            monthly_rows.append(monthly)

        if args.verbose:
            print(f"[{region}] {station_key}: cols={len(cols_str)} rows={len(df)}")

    daily_stats = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    monthly_stats = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    inventory = pd.DataFrame(inventory_rows)

    daily_path = tables / DAILY_OUT_NAME
    monthly_path = tables / MONTHLY_OUT_NAME
    inv_path = tables / INVENTORY_OUT_NAME

    if daily_path.exists() and not args.overwrite:
        print(f"SKIP daily exists: {daily_path}")
    else:
        tmp = daily_path.with_suffix(".parquet.tmp")
        daily_stats.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(daily_path)

    if monthly_path.exists() and not args.overwrite:
        print(f"SKIP monthly exists: {monthly_path}")
    else:
        tmp = monthly_path.with_suffix(".parquet.tmp")
        monthly_stats.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(monthly_path)

    if inv_path.exists() and not args.overwrite:
        print(f"SKIP inventory exists: {inv_path}")
    else:
        tmp = inv_path.with_suffix(".parquet.tmp")
        inventory.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(inv_path)

    print("============================================================")
    print("06C complete")
    print(f"daily_stats rows : {len(daily_stats):,}")
    print(f"monthly rows     : {len(monthly_stats):,}")
    print(f"inventory rows   : {len(inventory):,}")
    print(f"Wrote: {daily_path}")
    print(f"Wrote: {monthly_path}")
    print(f"Wrote: {inv_path}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
