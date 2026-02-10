#!/usr/bin/env python3
"""
05_build_epw_index_and_extract.py

Build EPW index + extract hourly dry-bulb temperature (DBT) into Parquet files.

UPDATED NAMING (requested):
- D-<DATASET>__DBT__F-<FREQ>__L-<LOCATION>.parquet
  e.g.
    D-TMY__DBT__F-HR__L-AB.parquet
    D-RCP__DBT__F-HR__L-AB.parquet
    D-CTI__DBT__F-HR__L-AB.parquet

- Index + parse log:
    D-<DATASET>__epw_index.json
    D-<DATASET>__parse_log.csv

DATASET SPLITTING:
- Even if you scan mixed roots, outputs are written separately per dataset:
    - TMY  (baseline): sources 00__italy_climate_onebuilding OR 01__italy_epw_all
    - RCP  (morphed):  source 02__italy_fwg_outputs
    - CTI  (cti):      source 01__italy_cti  OR filename contains "cti"

NO LEGACY OUTPUT:
- Removed/disabled dbt_rh_tidy.parquet. Only DBT is extracted.

Hourly outputs:
- With --regional:
    D-<DATASET>__DBT__F-HR__L-<REGION>.parquet
  Each parquet contains columns: DBT, rel_path, scenario, region, dataset
  Index is datetime.

Optional daily percentiles:
- With --daily-percentiles:
    D-<DATASET>__DBT__F-DDP__L-<REGION>.parquet
  (Still optional; your main daily stats should come from 06_precompute_derived_stats.py)

Usage examples:
  # Baseline only
  python 05_build_epw_index_and_extract.py --root data/01__italy_epw_all --out data/03__italy_all_epw_DBT_streamlit --regional

  # RCP only
  python 05_build_epw_index_and_extract.py --root data/02__italy_fwg_outputs --out data/03__italy_all_epw_DBT_streamlit --regional

  # CTI only
  python 05_build_epw_index_and_extract.py --root data/01__italy_cti --out data/03__italy_all_epw_DBT_streamlit --regional

  # Mixed roots -> outputs still split into three datasets (TMY/RCP/CTI)
  python 05_build_epw_index_and_extract.py --root data/02__italy_fwg_outputs --root data/01__italy_epw_all --root data/01__italy_cti --out data/03__italy_all_epw_DBT_streamlit --regional
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Constants / configuration
# -----------------------------------------------------------------------------

DEFAULT_OUT_DIR = "data/03__italy_all_epw_DBT_streamlit"

EPW_COLS = {
    "year": 0,
    "month": 1,
    "day": 2,
    "hour": 3,
    "minute": 4,
    "DBT": 6,
}

REGION_RE = re.compile(r"^[A-Z]{2}$")
STATION_RE = re.compile(r"[\._](\d{5,})(?:_|$)", re.IGNORECASE)
FWG_TMYX_RANGE_RE = re.compile(r"\.TMYx\.\d{4}-\d{4}$", re.IGNORECASE)

SOURCE_MARKERS = {
    "00__italy_climate_onebuilding",
    "01__italy_epw_all",
    "02__italy_fwg_outputs",
    "01__italy_cti",
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class EPWIndexRecord:
    dataset: str              # TMY / RCP / CTI
    scenario: str             # tmyx / tmyx_2009-2023 / rcp45_2050 / cti / unknown
    source: str               # marker folder
    region: str               # AB / BC / ...
    group: str                # folder group (FWG) or filename stem (baseline)
    rel_path: str             # project-relative (POSIX separators)
    abs_path: str             # full path on disk
    filename: str
    size_bytes: int
    mtime_iso: str
    header_location_line: str
    location_name: Optional[str] = None
    country: Optional[str] = None
    wmo: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    tz: Optional[float] = None
    elev: Optional[float] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_name(dataset: str, what: str, freq: Optional[str] = None, loc: Optional[str] = None, ext: str = "") -> str:
    """
    Construct filenames like:
      D-TMY__epw_index.json
      D-RCP__parse_log.csv
      D-CTI__DBT__F-HR__L-AB.parquet
    """
    parts = [f"D-{dataset}", what]
    if freq:
        parts.append(f"F-{freq}")
    if loc:
        parts.append(f"L-{loc}")
    return "__".join(parts) + ext


def dataset_from_source_and_filename(source: str, filename: str) -> str:
    """
    Decide dataset label:
      - CTI: source is 01__italy_cti OR filename contains 'cti'
      - RCP: source is 02__italy_fwg_outputs OR filename contains 'rcp'
      - TMY: otherwise for baseline sources
    """
    s = (source or "").lower()
    fn = (filename or "").lower()

    if "01__italy_cti" in s or "cti" in fn:
        return "CTI"
    if "02__italy_fwg_outputs" in s or "rcp" in fn or "_ensemble_" in fn:
        return "RCP"
    # baseline sources
    return "TMY"


def rel_to_project(p: Path, project_root: Optional[Path], fallback_root: Optional[Path]) -> str:
    """Project-relative path if possible, POSIX slashes always."""
    try:
        if project_root is not None:
            try:
                rel = p.resolve().relative_to(project_root.resolve())
                return str(rel).replace("\\", "/")
            except Exception:
                pass
        if fallback_root is not None:
            try:
                rel = p.resolve().relative_to(fallback_root.resolve())
                return str(rel).replace("\\", "/")
            except Exception:
                pass
        return str(p).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def scenario_from_filename(filename: str, source: str) -> str:
    """
    Parse scenario variant from filename.
    - CTI: 'cti'
    - RCP: rcp26_2050 / rcp45_2080 etc.
    - TMY: tmyx or tmyx_YYYY-YYYY
    """
    fn = (filename or "").lower()
    src = (source or "").lower()

    if "01__italy_cti" in src or "cti" in fn:
        return "cti"

    m_rcp = re.search(r"(rcp(?:26|45|85)[_\-]?(?:2050|2080))", fn)
    if m_rcp:
        return m_rcp.group(1).replace("-", "_")

    m_tmyx = re.search(r"tmyx\.(\d{4}-\d{4})", fn)
    if m_tmyx:
        return f"tmyx_{m_tmyx.group(1)}"
    if "tmyx" in fn:
        return "tmyx"

    return "unknown"


def parse_rel_source_region_group(rel_path: str, filename: str) -> Tuple[str, str, str]:
    """
    From a project-relative path, extract (source, region, group).
    Supports:
      - .../01__italy_epw_all/<REGION>/<file>.epw
      - .../00__italy_climate_onebuilding/epw/<REGION>/<file>.epw
      - .../02__italy_fwg_outputs/<REGION>/<GROUP>/<file>.epw
      - .../01__italy_cti/<REGION>/<file>.epw (or similar)

    group:
      - For FWG outputs: folder name under region (baseline group folder)
      - Otherwise: filename stem
    """
    rel_norm = rel_path.replace("\\", "/")
    parts = [p for p in rel_norm.split("/") if p]

    src_i = None
    for i, p in enumerate(parts):
        if p in SOURCE_MARKERS:
            src_i = i
            break

    source = parts[src_i] if src_i is not None else (parts[0] if parts else "unknown")

    # region
    region = "NA"
    if src_i is not None:
        if source == "00__italy_climate_onebuilding":
            # .../<marker>/epw/<REGION>/...
            if src_i + 2 < len(parts) and parts[src_i + 1].lower() == "epw":
                region = parts[src_i + 2]
        else:
            if src_i + 1 < len(parts):
                region = parts[src_i + 1]
    else:
        if len(parts) > 1:
            region = parts[1]

    region = (region or "NA").upper()
    if not REGION_RE.match(region):
        # best-effort recovery: find any 2-letter segment
        for seg in parts:
            if REGION_RE.match(seg.upper()):
                region = seg.upper()
                break

    # group
    if source == "02__italy_fwg_outputs":
        # .../<marker>/<REGION>/<GROUP>/<file>.epw
        group = parts[src_i + 2] if (src_i is not None and src_i + 2 < len(parts)) else Path(filename).stem
    else:
        group = Path(filename).stem

    return source, region, group


def read_epw_header(epw_path: Path) -> List[str]:
    text = epw_path.read_text(encoding="utf-8", errors="ignore")
    return text.splitlines()[:8]


def parse_location_from_header(
    line: str,
) -> Tuple[
    Optional[str], Optional[str], Optional[str],
    Optional[float], Optional[float], Optional[float], Optional[float]
]:
    parts = [p.strip() for p in (line or "").split(",")]
    if len(parts) < 10 or parts[0].upper() != "LOCATION":
        return (None, None, None, None, None, None, None)

    city = parts[1] if len(parts) > 1 else None
    country = parts[3] if len(parts) > 3 else None
    wmo = parts[5] if len(parts) > 5 else None

    def _to_float(x: str) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    lat = _to_float(parts[6]) if len(parts) > 6 else None
    lon = _to_float(parts[7]) if len(parts) > 7 else None
    tz = _to_float(parts[8]) if len(parts) > 8 else None
    elev = _to_float(parts[9]) if len(parts) > 9 else None

    return city, country, wmo, lat, lon, tz, elev


def iter_epw_files(roots: List[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.epw"):
            parts = [x for x in p.parts]
            if "02__italy_fwg_outputs" in parts:
                try:
                    i = parts.index("02__italy_fwg_outputs")
                    if i + 2 < len(parts):
                        group = parts[i + 2]
                        if FWG_TMYX_RANGE_RE.search(group):
                            continue
                except ValueError:
                    pass
            yield p


def parse_epw_to_hourly_df(epw_path: Path) -> pd.DataFrame:
    """
    Parse EPW to hourly DBT dataframe.
    Index: datetime
    Column: DBT
    """
    usecols = [0, 1, 2, 3, 4, EPW_COLS["DBT"]]
    names = ["year", "month", "day", "hour", "minute", "DBT"]

    df = pd.read_csv(
        epw_path,
        skiprows=8,
        header=None,
        usecols=usecols,
        names=names,
        dtype={"year": int, "month": int, "day": int, "hour": int, "minute": int},
        na_values=["", "NA", "NaN"],
        engine="c",
    )

    # EPW hour is 1-24 -> convert to 0-23
    df["hour"] = df["hour"].clip(1, 24) - 1
    # EPW minute often 60 -> treat as 0
    df["minute"] = df["minute"].replace({60: 0})

    dt = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"], minute=df["minute"]),
        errors="coerce",
    )
    df = df.drop(columns=["year", "month", "day", "hour", "minute"])
    df.index = dt
    df = df[~df.index.isna()]
    df.index.name = "datetime"

    df["DBT"] = pd.to_numeric(df["DBT"], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------

def write_index_json(index_records: List[EPWIndexRecord], out_dir: Path, dataset: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / make_name(dataset, "epw_index", ext=".json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in index_records], f, ensure_ascii=False, indent=2)
    return out_path


def write_parse_log(log_rows: List[Dict], out_dir: Path, dataset: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / make_name(dataset, "parse_log", ext=".csv")
    pd.DataFrame(log_rows).to_csv(out_path, index=False)
    return out_path


def write_regional_hourly(regional_data: Dict[str, List[pd.DataFrame]], out_dir: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for region, frames in regional_data.items():
        if not frames:
            continue
        df = pd.concat(frames, axis=0, ignore_index=False).sort_index()
        out_path = out_dir / make_name(dataset, "DBT", freq="HR", loc=region, ext=".parquet")
        df.to_parquet(out_path, index=True)


def compute_daily_percentiles_regional(regional_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Optional: per file daily mean/max/percentiles from hourly DBT.
    Output naming uses F-DDP (daily distribution/percentiles).
    """
    out: Dict[str, pd.DataFrame] = {}
    for region, frames in regional_data.items():
        if not frames:
            continue
        df = pd.concat(frames, axis=0, ignore_index=False)
        df = df.reset_index().rename(columns={"datetime": "dt"})
        df["date"] = df["dt"].dt.date
        g = df.groupby(["rel_path", "scenario", "date"], dropna=False)

        agg = g["DBT"].agg(["mean", "max"]).rename(columns={"mean": "DBT_mean", "max": "DBT_max"})
        p95 = g["DBT"].quantile(0.95).rename("DBT_max_p95")
        p975 = g["DBT"].quantile(0.975).rename("DBT_max_p975")
        p99 = g["DBT"].quantile(0.99).rename("DBT_max_p99")

        dd = pd.concat([agg, p95, p975, p99], axis=1).reset_index()
        dd["month"] = pd.to_datetime(dd["date"]).dt.month
        dd["day"] = pd.to_datetime(dd["date"]).dt.day
        out[region] = dd
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build EPW index and extract hourly DBT (temperature) data.")
    parser.add_argument("--root", action="append", required=True, help="Root folder(s) to scan for EPW files. Can be repeated.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory (index + parquets).")
    parser.add_argument("--regional", action="store_true", help="Write regional hourly files D-<DS>__DBT__F-HR__L-<REGION>.parquet")
    parser.add_argument("--daily-percentiles", action="store_true", help="Write optional daily percentiles D-<DS>__DBT__F-DDP__L-<REGION>.parquet")
    parser.add_argument("--project-root", default=None, help="Project root for project-relative paths (optional).")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")

    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.root]
    out_dir = Path(args.out).resolve()

    project_root = Path(args.project_root).resolve() if args.project_root else None
    if project_root is None:
        try:
            project_root = Path(os.path.commonpath([str(r) for r in roots])).resolve()
        except Exception:
            project_root = roots[0].parent.resolve() if roots else Path(".").resolve()

    fallback_root = project_root

    # Per-dataset structures
    index_by_ds: Dict[str, List[EPWIndexRecord]] = defaultdict(list)
    log_by_ds: Dict[str, List[Dict]] = defaultdict(list)
    regional_by_ds: Dict[str, Dict[str, List[pd.DataFrame]]] = defaultdict(lambda: defaultdict(list))

    all_epws = list(iter_epw_files(roots))
    total = len(all_epws)
    ok = 0
    fail = 0

    if args.verbose:
        print(f"Discovered {total} EPW files across {len(roots)} roots.")
        for r in roots:
            print(f" - {r}")

    for i, p in enumerate(all_epws, start=1):
        try:
            st_ = p.stat()
            rel = rel_to_project(p, project_root=project_root, fallback_root=fallback_root)

            source, region, group = parse_rel_source_region_group(rel, p.name)
            scenario = scenario_from_filename(p.name, source)
            dataset = dataset_from_source_and_filename(source, p.name)

            header = read_epw_header(p)
            loc_line = header[0] if header else ""
            city, country, wmo, lat, lon, tz, elev = parse_location_from_header(loc_line)

            index_by_ds[dataset].append(
                EPWIndexRecord(
                    dataset=dataset,
                    scenario=scenario,
                    source=source,
                    region=region,
                    group=group,
                    rel_path=rel,
                    abs_path=str(p.resolve()),
                    filename=p.name,
                    size_bytes=st_.st_size,
                    mtime_iso=pd.to_datetime(st_.st_mtime, unit="s").isoformat(),
                    header_location_line=loc_line,
                    location_name=city,
                    country=country,
                    wmo=wmo,
                    lat=lat,
                    lon=lon,
                    tz=tz,
                    elev=elev,
                )
            )

            # Hourly extraction (DBT only)
            if args.regional or args.daily_percentiles:
                dfh = parse_epw_to_hourly_df(p)
                # metadata columns
                dfh["rel_path"] = rel
                dfh["scenario"] = scenario
                dfh["region"] = region
                dfh["dataset"] = dataset
                regional_by_ds[dataset][region].append(dfh)

            ok += 1
            log_by_ds[dataset].append({"rel_path": rel, "status": "OK", "error": ""})

            if args.verbose and (i % 50 == 0 or i == total):
                print(f"[{i}/{total}] OK={ok} FAIL={fail}")

        except Exception as e:
            fail += 1
            rel = rel_to_project(p, project_root=project_root, fallback_root=fallback_root)
            # if we can infer dataset, log it; else use "UNK"
            try:
                source, _, _ = parse_rel_source_region_group(rel, p.name)
                ds = dataset_from_source_and_filename(source, p.name)
            except Exception:
                ds = "UNK"
            log_by_ds[ds].append({"rel_path": rel, "status": "FAIL", "error": f"{type(e).__name__}: {e}"})
            if args.verbose:
                print(f"[{i}/{total}] FAIL: {p} -> {e}")
                traceback.print_exc()

    # Write outputs per dataset
    for ds, idx_rows in index_by_ds.items():
        write_index_json(idx_rows, out_dir, ds)
    for ds, log_rows in log_by_ds.items():
        write_parse_log(log_rows, out_dir, ds)

    if args.regional:
        for ds, reg_map in regional_by_ds.items():
            write_regional_hourly(reg_map, out_dir, ds)

    if args.daily_percentiles:
        for ds, reg_map in regional_by_ds.items():
            daily_by_region = compute_daily_percentiles_regional(reg_map)
            for region, dd in daily_by_region.items():
                out_path = out_dir / make_name(ds, "DBT", freq="DDP", loc=region, ext=".parquet")
                dd.to_parquet(out_path, index=False)

    print(f"Done. EPWs: {total} | OK={ok} | FAIL={fail}")
    print(f"Wrote outputs per dataset into: {out_dir}")
    if args.regional:
        print("Hourly files pattern: D-<DS>__DBT__F-HR__L-<REGION>.parquet")
    if args.daily_percentiles:
        print("Daily percentiles pattern: D-<DS>__DBT__F-DDP__L-<REGION>.parquet")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
