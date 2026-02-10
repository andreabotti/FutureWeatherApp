#!/usr/bin/env python3
"""
05C_build_cti_station_parquets.py

Build one parquet per CTI station containing:
- single scenario: cti

Input root:
  data/01__italy_cti/
  - preferred layout: <CTI_ROOT>/<REGION>/*.epw
  - fallback:         <CTI_ROOT>/**/*.epw

Output:
  data/04__italy_cti_parquet/<REGION>/<STATION_KEY>.parquet

STATION_KEY is derived from filename stem when possible; otherwise
fallback to EPW header metadata (location name + WMO).

DBT only. No partitioned writes, no _tmp_parts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

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
    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(8):
            next(f, None)
        rows = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                m = int(parts[1])
                d = int(parts[2])
                h = int(parts[3])
                dbt = float(parts[6])
            except Exception:
                continue
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
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


# ----------------------------
# Station key + region parsing
# ----------------------------
FILENAME_WITH_WMO_RE = re.compile(r"^(?P<key>.+\.\d{6})$")
ITA_REGION_RE = re.compile(r"ITA_(?P<region>[A-Z]{2})_", re.IGNORECASE)
CTI_FILENAME_RE = re.compile(r"^(?P<region>[A-Z]{2})__(?P<province>[A-Z]{2})__(?P<station>.+)$")


def _sanitize_token(value: str) -> str:
    out = str(value or "").strip()
    out = re.sub(r"\s+", "_", out)
    out = re.sub(r"[^\w\.\-]+", "_", out)
    out = re.sub(r"_+", "_", out)
    return out.strip("_")


def _parse_epw_header_location(epw_path: Path) -> dict[str, Optional[str]]:
    """
    Parse EPW LOCATION header:
    LOCATION,city,state,country,source,WMO,latitude,longitude,timezone,elevation
    """
    try:
        with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
            line = next(f, "")
    except Exception:
        return {}
    if not line.startswith("LOCATION"):
        return {}
    parts = [p.strip() for p in line.split(",")]
    city = parts[1] if len(parts) > 1 else None
    wmo = parts[5] if len(parts) > 5 else None
    wmo = wmo if wmo and wmo.isdigit() else None
    return {"city": city, "wmo": wmo}


def derive_station_key(epw_path: Path) -> str:
    stem = epw_path.stem.strip()
    if not stem:
        stem = epw_path.name

    if FILENAME_WITH_WMO_RE.match(stem):
        return stem
    if stem.upper().startswith("ITA_"):
        return stem

    m = CTI_FILENAME_RE.match(stem)
    if m:
        region = m.group("region").upper()
        station = m.group("station")
        station = re.sub(r"\.\d+$", "", station)
        station = _sanitize_token(station)
        return f"ITA_{region}_{station}"

    meta = _parse_epw_header_location(epw_path)
    station = _sanitize_token(meta.get("city") or stem)
    wmo = meta.get("wmo")
    if wmo:
        return f"{station}.{wmo}"
    return station


def derive_region(epw_path: Path, region_from_dir: Optional[str]) -> str:
    if region_from_dir:
        return str(region_from_dir)
    m = ITA_REGION_RE.search(epw_path.stem)
    if m:
        return m.group("region").upper()
    return "NA"


def discover_cti_epws(root: Path) -> Dict[str, List[Path]]:
    """
    Preferred layout: <CTI_ROOT>/<REGION>/*.epw
    Fallback:         <CTI_ROOT>/**/*.epw (region inferred from filename or NA)
    """
    out: Dict[str, List[Path]] = {}
    region_dirs = [p for p in root.iterdir() if p.is_dir()]
    for region_dir in sorted(region_dirs):
        files = sorted([p for p in region_dir.glob("*.epw") if p.is_file()])
        if files:
            out[region_dir.name] = files

    if out:
        return out

    files = sorted([p for p in root.rglob("*.epw") if p.is_file()])
    for p in files:
        region = derive_region(p, None)
        out.setdefault(region, []).append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cti-root", required=True, help="data/01__italy_cti")
    ap.add_argument("--out-root", required=True, help="data/04__italy_cti_parquet")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fixed-year", type=int, default=2001)
    args = ap.parse_args()

    cti_root = Path(args.cti_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files_by_region = discover_cti_epws(cti_root)
    regions = sorted(files_by_region.keys())
    total_written = 0
    total_series = 0
    total_errors = 0

    if args.verbose:
        n_files = sum(len(v) for v in files_by_region.values())
        print(f"[discover] regions={len(regions)} files={n_files}")

    for region in regions:
        out_region = out_root / region
        out_region.mkdir(parents=True, exist_ok=True)

        station_frames: Dict[str, Dict[str, pd.Series]] = {}
        for p in files_by_region.get(region, []):
            station_key = derive_station_key(p)
            if not station_key:
                if args.verbose:
                    print(f"[{region}] SKIP (no station key): {p.name}")
                continue
            try:
                s = read_epw_dbt(p, fixed_year=args.fixed_year)
                s.name = "cti"
                if station_key in station_frames and "cti" in station_frames[station_key]:
                    if args.verbose:
                        print(f"[{region}] DUPLICATE cti for {station_key}: {p.name}")
                    continue
                station_frames.setdefault(station_key, {})["cti"] = s
                total_series += 1
            except Exception as e:
                total_errors += 1
                if args.verbose:
                    print(f"[{region}] ERROR {p.name}: {e}")

        for station_key, series_map in station_frames.items():
            if not series_map:
                continue
            df = pd.DataFrame(series_map)
            df.index.name = "dt"
            df = df.sort_index()

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
    print("05C complete")
    print(f"Stations written : {total_written}")
    print(f"Series ingested  : {total_series}")
    print(f"Errors           : {total_errors}")
    print(f"Output root      : {out_root}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
