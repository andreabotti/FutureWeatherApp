#!/usr/bin/env python3
"""
05B_build_station_parquets.py

Build one parquet per station containing:
- baseline variants (tmyx, tmyx_2004-2018, tmyx_2007-2021, tmyx_2009-2023)
- future variants per baseline variant (e.g. tmyx__rcp26_2050, tmyx_2007-2021__rcp45_2080, ...)

Inputs (flat per region):
  data/01__italy_tmy_dbt/<REGION>/*.epw
  data/03__italy_tmy_fwg/<REGION>/*.epw   (renamed as baselineName__Ensemble__rcp26__2050.epw)

Output:
  data/04__italy_tmy_fwg_parquet/<REGION>/<STATION_KEY>.parquet

STATION_KEY is parsed as:
  <everything before "_TMYx">  (e.g. ITA_AB_Fucino.162270)

All series are DBT only.

No partitioned writes, no _tmp_parts.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd


# ----------------------------
# EPW parsing (DBT only)
# ----------------------------
def read_epw_dbt(epw_path: Path, fixed_year: int = 2001) -> pd.Series:
    """
    Read EPW and return a Series of DBT (°C) indexed by hourly datetime.
    Uses fixed_year for consistent merging across files.
    EPW columns: Year,Month,Day,Hour,Minute,Flags,DryBulb,... -> DBT is column index 6 (0-based).
    """
    # EPW: first 8 lines are header/meta
    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(8):
            next(f, None)
        rows = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                m = int(parts[1]); d = int(parts[2]); h = int(parts[3])
                dbt = float(parts[6])
            except Exception:
                continue
            # EPW hour is 1-24; map to 0-23
            hh = h - 1 if 1 <= h <= 24 else h
            rows.append((m, d, hh, dbt))

    if not rows:
        raise ValueError(f"No EPW data rows parsed: {epw_path}")

    df = pd.DataFrame(rows, columns=["month", "day", "hour", "dbt"])
    dt = pd.to_datetime(
        {
            "year": fixed_year,
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
        },
        errors="coerce",
    )
    s = pd.Series(df["dbt"].values, index=dt, name=epw_path.stem)
    s = s[~s.index.isna()]
    # Ensure sorted and unique
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


# ----------------------------
# Filename parsing
# ----------------------------
BASELINE_RE = re.compile(r"^(?P<prefix>.+?)_TMYx(?P<suffix>(?:\.\d{4}-\d{4})?)$", re.IGNORECASE)
FWG_RE = re.compile(
    r"^(?P<baseline>.+?)__Ensemble__(?P<rcp>rcp\d{2})__(?P<year>\d{4})$",
    re.IGNORECASE,
)

def parse_station_key_and_baseline_variant_from_baseline_filename(stem: str) -> Optional[Tuple[str, str]]:
    """
    Baseline EPW stems look like:
      ITA_AB_Fucino.162270_TMYx
      ITA_AB_Fucino.162270_TMYx.2007-2021
    Return: (station_key, baseline_variant)
      station_key = 'ITA_AB_Fucino.162270'
      baseline_variant = 'tmyx' or 'tmyx_2007-2021'
    """
    m = BASELINE_RE.match(stem)
    if not m:
        return None
    station_key = m.group("prefix")
    suffix = m.group("suffix") or ""
    if suffix.startswith("."):
        suffix = suffix[1:]
    baseline_variant = "tmyx" if not suffix else f"tmyx_{suffix}"
    return station_key, baseline_variant

def parse_station_key_and_future_variant_from_fwg_filename(stem: str) -> Optional[Tuple[str, str]]:
    """
    FWG stems look like:
      <baselineStem>__Ensemble__rcp26__2050
    where <baselineStem> is exactly a baseline folder/name like:
      ITA_AB_Fucino.162270_TMYx.2007-2021
    Return: (station_key, future_variant)
      station_key = 'ITA_AB_Fucino.162270'
      future_variant = '<baseline_variant>__rcp26_2050'
    """
    m = FWG_RE.match(stem)
    if not m:
        return None
    baseline_stem = m.group("baseline")
    rcp = m.group("rcp").lower()
    year = m.group("year")
    parsed = parse_station_key_and_baseline_variant_from_baseline_filename(baseline_stem)
    if not parsed:
        return None
    station_key, baseline_variant = parsed
    future_variant = f"{baseline_variant}__{rcp}_{year}"
    return station_key, future_variant


def discover_files_by_region(root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for region_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        region = region_dir.name
        out[region] = sorted([p for p in region_dir.glob("*.epw") if p.is_file()])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmy-root", required=True, help="data/01__italy_tmy_dbt")
    ap.add_argument("--fwg-root", required=True, help="data/03__italy_tmy_fwg")
    ap.add_argument("--out-root", required=True, help="data/04__italy_tmy_fwg_parquet")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fixed-year", type=int, default=2001)
    args = ap.parse_args()

    tmy_root = Path(args.tmy_root).resolve()
    fwg_root = Path(args.fwg_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tmy_by_region = discover_files_by_region(tmy_root)
    fwg_by_region = discover_files_by_region(fwg_root)

    regions = sorted(set(tmy_by_region.keys()) | set(fwg_by_region.keys()))
    total_written = 0
    total_series = 0
    total_errors = 0

    for region in regions:
        out_region = out_root / region
        out_region.mkdir(parents=True, exist_ok=True)

        # Collect series per station
        station_frames: Dict[str, Dict[str, pd.Series]] = {}

        # --- Baselines
        for p in tmy_by_region.get(region, []):
            parsed = parse_station_key_and_baseline_variant_from_baseline_filename(p.stem)
            if not parsed:
                if args.verbose:
                    print(f"[{region}] SKIP baseline (unparsed): {p.name}")
                continue
            station_key, baseline_variant = parsed
            try:
                s = read_epw_dbt(p, fixed_year=args.fixed_year)
                s.name = baseline_variant
                station_frames.setdefault(station_key, {})[baseline_variant] = s
                total_series += 1
            except Exception as e:
                total_errors += 1
                if args.verbose:
                    print(f"[{region}] ERROR baseline {p.name}: {e}")

        # --- Futures
        for p in fwg_by_region.get(region, []):
            parsed = parse_station_key_and_future_variant_from_fwg_filename(p.stem)
            if not parsed:
                if args.verbose:
                    print(f"[{region}] SKIP fwg (unparsed): {p.name}")
                continue
            station_key, future_variant = parsed
            try:
                s = read_epw_dbt(p, fixed_year=args.fixed_year)
                s.name = future_variant
                station_frames.setdefault(station_key, {})[future_variant] = s
                total_series += 1
            except Exception as e:
                total_errors += 1
                if args.verbose:
                    print(f"[{region}] ERROR fwg {p.name}: {e}")

        # --- Write one parquet per station
        for station_key, series_map in station_frames.items():
            if not series_map:
                continue

            df = pd.DataFrame(series_map)
            df.index.name = "dt"
            df = df.sort_index()

            # keep only full-year hour rows if present; but don't enforce if source differs
            out_path = out_region / f"{station_key}.parquet"
            if out_path.exists() and not args.overwrite:
                if args.verbose:
                    print(f"[{region}] SKIP exists: {out_path.name}")
                continue

            tmp = out_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, engine="pyarrow", index=True)
            tmp.replace(out_path)

            total_written += 1
            if args.verbose:
                print(f"[{region}] WROTE {out_path.name} | cols={len(df.columns)} rows={len(df)}")

    print("============================================================")
    print("05B complete")
    print(f"Stations written : {total_written}")
    print(f"Series ingested  : {total_series}")
    print(f"Errors           : {total_errors}")
    print(f"Output root      : {out_root}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
