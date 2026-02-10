#!/usr/bin/env python3
"""
06_precompute_derived_stats.py

Rebuild derived (daily) statistics and downstream aggregations from hourly DBT 
parquet files produced by 05_build_epw_index_and_extract.py (new naming rules).

This script expects hourly files like:
  D-TMY__DBT__F-HR__L-AB.parquet
  D-RCP__DBT__F-HR__L-AB.parquet
  D-CTI__DBT__F-HR__L-AB.parquet

It produces:
1) Combined daily stats:
   D-ALL__DBT__F-DD__L-ALL.parquet
   daily_stats.parquet (backwards compat copy)

2) Per-dataset daily stats:
   D-TMY__DBT__F-DD__L-ALL.parquet
   D-RCP__DBT__F-DD__L-ALL.parquet
   D-CTI__DBT__F-DD__L-ALL.parquet

3) Derived aggregations (per percentile, e.g. P-99), computed from HOURLY DBT
   (percentiles from hourly values = more robust than from daily max):
   D-ALL__FileStats__P-99.parquet
   D-ALL__LocationDeltas__P-99.parquet
   D-ALL__LocationStats__P-99.parquet
   D-ALL__MonthlyDeltas__F-MM__P-99.parquet

Daily stats schema:
  dataset, region, rel_path, scenario, month, day, DBT_mean, DBT_max

Notes:
- This script does NOT attempt to process CTI CSVs. CTI should already be present
  as EPW-derived parquet from script 05 (dataset CTI).
- This script does NOT write any legacy "dbt_rh_tidy.parquet" (DBT only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Filename patterns (new convention)
# ---------------------------------------------------------------------

# Hourly files created by 05:
#   D-<DATASET>__DBT__F-HR__L-<REGION>.parquet
HOURLY_RE = re.compile(r"^D-(?P<dataset>[^_]+)__DBT__F-HR__L-(?P<loc>.+)\.parquet$", re.IGNORECASE)

# Per-dataset daily outputs:
#   D-<DATASET>__DBT__F-DD__L-ALL.parquet
def daily_out_name(dataset: str) -> str:
    return f"D-{dataset}__DBT__F-DD__L-ALL.parquet"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@dataclass
class HourlyFileInfo:
    path: Path
    dataset: str
    region: str


def discover_hourly_files(data_dir: Path) -> List[HourlyFileInfo]:
    """
    Find hourly parquet files following the new naming convention.
    """
    infos: List[HourlyFileInfo] = []
    for p in sorted(data_dir.glob("D-*__DBT__F-HR__L-*.parquet")):
        m = HOURLY_RE.match(p.name)
        if not m:
            continue
        dataset = (m.group("dataset") or "").upper()
        region = (m.group("loc") or "").upper()
        infos.append(HourlyFileInfo(path=p, dataset=dataset, region=region))
    return infos


def load_hourly(path: Path, dataset: str, region: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load a single hourly parquet and enforce required columns.

    Required:
      - datetime index (or 'datetime' column)
      - DBT
      - rel_path
      - scenario

    Adds/normalizes:
      - dataset
      - region
    """
    df = pd.read_parquet(path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        else:
            # Try to coerce existing index
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
            except Exception as e:
                raise ValueError(f"{path.name}: could not interpret datetime index and no 'datetime' column found") from e

    # Required columns
    required = ["DBT", "rel_path", "scenario"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Make sure script 05 writes DBT, rel_path, scenario."
        )

    # Add / normalize dataset + region
    df["dataset"] = str(dataset).upper()
    if "region" not in df.columns:
        df["region"] = str(region).upper()
    else:
        df["region"] = df["region"].astype(str).str.upper().fillna(str(region).upper())

    # Clean DBT
    df["DBT"] = pd.to_numeric(df["DBT"], errors="coerce")

    if verbose:
        n_files = df["rel_path"].nunique()
        sc = df["scenario"].astype(str).unique().tolist()
        print(f"  Loaded {path.name}: {len(df):,} rows | files={n_files:,} | scenarios={sc[:8]}{'...' if len(sc)>8 else ''}")

    return df


def compute_daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily mean and max by:
      dataset, region, rel_path, scenario, month, day
    """
    if hourly.empty:
        return pd.DataFrame(columns=["dataset", "region", "rel_path", "scenario", "month", "day", "DBT_mean", "DBT_max"])

    # Build a working df with datetime column
    df = hourly.reset_index().rename(columns={"index": "datetime"})
    df = df.rename(columns={df.columns[0]: "datetime"}) if "datetime" not in df.columns else df

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "DBT", "rel_path", "scenario", "dataset", "region"])
    if df.empty:
        return pd.DataFrame(columns=["dataset", "region", "rel_path", "scenario", "month", "day", "DBT_mean", "DBT_max"])

    df["month"] = df["datetime"].dt.month.astype("int8")
    df["day"] = df["datetime"].dt.day.astype("int8")

    daily = (
        df.groupby(["dataset", "region", "rel_path", "scenario", "month", "day"], as_index=False, observed=True)
          .agg(
              DBT_mean=("DBT", "mean"),
              DBT_max=("DBT", "max"),
          )
    )

    # dtypes for size/perf
    daily["dataset"] = daily["dataset"].astype("category")
    daily["region"] = daily["region"].astype("category")
    daily["rel_path"] = daily["rel_path"].astype("category")
    daily["scenario"] = daily["scenario"].astype("category")
    daily["DBT_mean"] = daily["DBT_mean"].astype("float32")
    daily["DBT_max"] = daily["DBT_max"].astype("float32")

    return daily


def load_index_from_dir(data_dir: Path) -> pd.DataFrame:
    """Load index from D-*__epw_index.json files in the directory."""
    json_files = sorted(data_dir.glob("D-*__epw_index.json"))
    if not json_files:
        legacy = data_dir / "epw_index.json"
        if legacy.exists():
            json_files = [legacy]
    
    if not json_files:
        return pd.DataFrame()
    
    records = []
    for p in json_files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
    
    if not records:
        return pd.DataFrame()
    
    idx = pd.DataFrame(records)
    
    # Add variant if missing
    if "variant" not in idx.columns and "filename" in idx.columns:
        idx["variant"] = idx["filename"].apply(lambda fn: Path(fn).stem.split("_")[0] if isinstance(fn, str) else "")

    # Ensure group_key exists (prefer FWG folder group from rel_path)
    def group_key_from_rel_path(rel_path: str, filename: str) -> str:
        rel_norm = str(rel_path or "").replace("\\", "/")
        parts = [p for p in rel_norm.split("/") if p]
        try:
            i = parts.index("02__italy_fwg_outputs")
            if i + 2 < len(parts):
                return parts[i + 2]
        except ValueError:
            pass
        return Path(filename).stem if isinstance(filename, str) else ""

    if "group_key" not in idx.columns:
        group_vals = []
        for g, rp, fn in zip(
            idx.get("group", [None] * len(idx)),
            idx.get("rel_path", [""] * len(idx)),
            idx.get("filename", [""] * len(idx)),
        ):
            if isinstance(g, str) and g:
                group_vals.append(Path(g).stem if g.lower().endswith(".epw") else g)
            else:
                group_vals.append(group_key_from_rel_path(rp, fn))
        idx["group_key"] = group_vals
    else:
        # normalize empties
        idx["group_key"] = [
            (g if isinstance(g, str) and g else group_key_from_rel_path(rp, fn))
            for g, rp, fn in zip(
                idx.get("group_key", [None] * len(idx)),
                idx.get("rel_path", [""] * len(idx)),
                idx.get("filename", [""] * len(idx)),
            )
        ]
    
    # Derive location_id (same logic as app's f10__load_index)
    if "location_id" not in idx.columns:
        def parse_station_id_from_filename(fn: str) -> str:
            """Extract station ID from filename (e.g., ITA_Abruzzo_Chieti.162270_.epw -> 162270)."""
            if not isinstance(fn, str):
                return ""
            stem = Path(fn).stem
            parts = stem.split(".")
            for p in parts:
                # Station ID pattern: 6 digits, possibly ending with underscore
                clean = p.strip("_")
                if len(clean) == 6 and clean.isdigit():
                    return clean
            return ""
        
        def parse_station_id_from_group_key(gk: str) -> str:
            """Extract station ID from group_key (e.g., ITA_Abruzzo_Chieti.162270_ -> 162270)."""
            if not isinstance(gk, str):
                return ""
            parts = gk.split(".")
            for p in parts:
                clean = p.strip("_")
                if len(clean) == 6 and clean.isdigit():
                    return clean
            return ""
        
        # location_id priority: wmo → station_id from filename → group_station_id → filename stem → scenario
        wmo_s = idx.get("wmo", pd.Series([pd.NA] * len(idx))).astype(str).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        station_id_fn = idx.get("filename", pd.Series([""] * len(idx))).apply(parse_station_id_from_filename)
        
        # group_key or group
        group_key_s = idx.get("group_key", idx.get("group", pd.Series([pd.NA] * len(idx))))
        group_station_id = group_key_s.astype(str).apply(parse_station_id_from_group_key)
        
        filename_stem = idx.get("filename", pd.Series([""] * len(idx))).apply(
            lambda fn: Path(fn).stem if isinstance(fn, str) else pd.NA
        )
        scenario_s = idx.get("scenario", pd.Series([pd.NA] * len(idx))).astype(str)
        
        idx["location_id"] = (
            wmo_s.fillna(station_id_fn).fillna(group_station_id).fillna(filename_stem).fillna(scenario_s).astype(str)
        )
    
    # Derive is_cti flag
    if "is_cti" not in idx.columns:
        dataset_s = idx.get("dataset", pd.Series([""] * len(idx))).astype(str).str.upper()
        source_s = idx.get("source", pd.Series([""] * len(idx))).astype(str)
        scenario_s = idx.get("scenario", pd.Series([""] * len(idx))).astype(str)
        idx["is_cti"] = (
            dataset_s.eq("CTI")
            | source_s.str.contains("cti", case=False, na=False)
            | scenario_s.str.contains("cti", case=False, na=False)
        )
    
    return idx


def compute_derived_stats_from_hourly(
    hourly: pd.DataFrame,
    idx: pd.DataFrame,
    data_dir: Path,
    percentiles: List[int],
    verbose: bool = False,
) -> None:
    """
    Compute and save derived aggregations from hourly DBT (percentiles from hourly, more robust).
    
    Uses: f123h, f28bh, f125h, f124h (hourly-based stats).
    """
    script_path = Path(__file__).resolve()
    fwg_root = script_path.parent.parent.parent
    libs_path = fwg_root / "libs"
    
    if not libs_path.exists():
        print(f"⚠️  Could not find libs folder at {libs_path}")
        return
    
    sys.path.insert(0, str(libs_path))
    try:
        import fn__libs as h
    except ImportError as e:
        print(f"⚠️  Could not import fn__libs: {e}")
        return
    
    if hourly.empty or idx.empty:
        print("⚠️  Hourly data or index is empty - skipping derived stats")
        return
    
    variants = sorted([v for v in idx["variant"].dropna().unique() if str(v).lower() != "cti"])
    if not variants:
        print("⚠️  No variants found in index - skipping derived stats")
        return
    
    print("\n" + "=" * 72)
    print("Computing derived aggregations from HOURLY data (percentiles from hourly)...")
    print("=" * 72)
    
    for percentile_pct in percentiles:
        percentile_dec = percentile_pct / 100.0
        print(f"\nPercentile: {percentile_pct}% (={percentile_dec:.2f})")
        
        t0 = time.time()
        try:
            file_stats = h.f123h__build_file_stats_from_hourly(hourly, percentile_dec)
            if not file_stats.empty:
                file_stats["percentile"] = percentile_dec
                out_path = data_dir / f"D-ALL__FileStats__P-{percentile_pct}.parquet"
                file_stats.to_parquet(out_path, index=False)
                print(f"   ✓ FileStats: {out_path.name} ({len(file_stats):,} rows, {time.time()-t0:.2f}s)")
            else:
                print(f"   ⚠️  FileStats is empty (skipped)")
        except Exception as e:
            print(f"   ❌ FileStats failed: {e}")
        
        location_stats_parts = []
        for variant in variants:
            t0 = time.time()
            try:
                loc_stats = h.f28bh__compute_location_stats_for_variant_from_hourly(
                    hourly, idx, variant=variant, percentile=percentile_dec
                )
                if not loc_stats.empty:
                    loc_stats["variant"] = variant
                    loc_stats["percentile"] = percentile_dec
                    location_stats_parts.append(loc_stats)
                    if verbose:
                        print(f"      - {variant}: {len(loc_stats):,} locations ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"      ❌ {variant}: {e}")
        
        if location_stats_parts:
            location_stats_all = pd.concat(location_stats_parts, ignore_index=True)
            out_path = data_dir / f"D-ALL__LocationStats__P-{percentile_pct}.parquet"
            location_stats_all.to_parquet(out_path, index=False)
            print(f"   ✓ LocationStats: {out_path.name} ({len(location_stats_all):,} rows, {len(variants)} variants)")
        else:
            print(f"   ⚠️  LocationStats is empty (no variants)")
        
        baseline_variants = [v for v in variants if "tmy" in str(v).lower()]
        compare_variants = [v for v in variants if "rcp" in str(v).lower() or "ssp" in str(v).lower()]
        if not baseline_variants:
            baseline_variants = variants[:1] if variants else []
        if not compare_variants:
            compare_variants = variants[1:] if len(variants) > 1 else []
        
        location_deltas_parts = []
        pair_count = 0
        for baseline in baseline_variants:
            for compare in compare_variants:
                t0 = time.time()
                try:
                    deltas = h.f125h__compute_location_deltas_from_hourly(
                        hourly, idx,
                        baseline_variant=baseline,
                        compare_variant=compare,
                        percentile=percentile_dec,
                        verbose=verbose,
                    )
                    if not deltas.empty:
                        deltas["baseline_variant"] = baseline
                        deltas["compare_variant"] = compare
                        deltas["percentile"] = percentile_dec
                        location_deltas_parts.append(deltas)
                        pair_count += 1
                        if verbose:
                            print(f"      - {baseline} vs {compare}: {len(deltas):,} locations ({time.time()-t0:.2f}s)")
                except Exception as e:
                    if verbose:
                        print(f"      ❌ {baseline} vs {compare}: {e}")
        
        if location_deltas_parts:
            location_deltas_all = pd.concat(location_deltas_parts, ignore_index=True)
            out_path = data_dir / f"D-ALL__LocationDeltas__P-{percentile_pct}.parquet"
            location_deltas_all.to_parquet(out_path, index=False)
            print(f"   ✓ LocationDeltas: {out_path.name} ({len(location_deltas_all):,} rows, {pair_count} pairs)")
        else:
            print(f"   ⚠️  LocationDeltas is empty (no pairs)")
        
        monthly_deltas_parts = []
        for baseline in baseline_variants:
            for compare in compare_variants:
                for metric_key in ["dTmax", "dTavg"]:
                    t0 = time.time()
                    try:
                        monthly = h.f124h__build_monthly_delta_table_from_hourly(
                            hourly, idx,
                            baseline_variant=baseline,
                            compare_variant=compare,
                            percentile=percentile_dec,
                            metric_key=metric_key,
                            verbose=verbose,
                        )
                        if not monthly.empty:
                            monthly["baseline_variant"] = baseline
                            monthly["compare_variant"] = compare
                            monthly["metric"] = metric_key
                            monthly["percentile"] = percentile_dec
                            monthly_deltas_parts.append(monthly)
                            if verbose:
                                print(f"      - {baseline} vs {compare} ({metric_key}): {len(monthly):,} locations ({time.time()-t0:.2f}s)")
                    except Exception as e:
                        if verbose:
                            print(f"      ❌ {baseline} vs {compare} ({metric_key}): {e}")
        
        if monthly_deltas_parts:
            monthly_deltas_all = pd.concat(monthly_deltas_parts, ignore_index=True)
            out_path = data_dir / f"D-ALL__MonthlyDeltas__F-MM__P-{percentile_pct}.parquet"
            monthly_deltas_all.to_parquet(out_path, index=False)
            print(f"   ✓ MonthlyDeltas: {out_path.name} ({len(monthly_deltas_all):,} rows)")
        else:
            print(f"   ⚠️  MonthlyDeltas is empty")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute derived daily stats and aggregations from hourly DBT parquet files.")
    ap.add_argument(
        "--data-dir",
        type=str,
        default="data/03__italy_all_epw_DBT_streamlit",
        help="Directory containing hourly parquet files (default: data/03__italy_all_epw_DBT_streamlit).",
    )
    ap.add_argument(
        "--write-per-dataset",
        action="store_true",
        help="Also write per-dataset daily parquet files (D-<DS>__DBT__F-DD__L-ALL.parquet).",
    )
    ap.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[95, 97, 99],
        help="Percentiles to compute for derived stats (default: 95 97 99).",
    )
    ap.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip computing derived aggregations (FileStats, LocationDeltas, etc.).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run script 05 first (it will create the folder and hourly files).")
        sys.exit(1)

    hourly_infos = discover_hourly_files(data_dir)
    if not hourly_infos:
        print(f"❌ No hourly files found in {data_dir}")
        print("   Expected pattern: D-<DATASET>__DBT__F-HR__L-<REGION>.parquet")
        sys.exit(1)

    if args.verbose:
        print(f"\nData directory: {data_dir}")
        print(f"Discovered hourly files: {len(hourly_infos)}")
        for inf in hourly_infos[:12]:
            print(f"  - {inf.path.name}")

    # Load hourly and compute daily per dataset; keep hourly for robust percentile stats
    daily_by_dataset: Dict[str, List[pd.DataFrame]] = {}
    hourly_parts: List[pd.DataFrame] = []
    for info in hourly_infos:
        dfh = load_hourly(info.path, dataset=info.dataset, region=info.region, verbose=args.verbose)
        daily = compute_daily_from_hourly(dfh)
        daily_by_dataset.setdefault(info.dataset, []).append(daily)
        hourly_parts.append(dfh)

    # Combine per dataset frames
    daily_ds_frames: Dict[str, pd.DataFrame] = {}
    for ds, parts in daily_by_dataset.items():
        if not parts:
            continue
        daily_ds_frames[ds] = pd.concat(parts, axis=0, ignore_index=True)

    # Combined daily stats for app compatibility
    combined_daily = pd.concat(list(daily_ds_frames.values()), axis=0, ignore_index=True)

    # New naming: D-ALL__DBT__F-DD__L-ALL.parquet
    out_combined_new = data_dir / "D-ALL__DBT__F-DD__L-ALL.parquet"
    out_combined_new.parent.mkdir(parents=True, exist_ok=True)
    combined_daily.to_parquet(out_combined_new, index=False)
    
    # Backwards compat: daily_stats.parquet (copy)
    out_combined_legacy = data_dir / "daily_stats.parquet"
    combined_daily.to_parquet(out_combined_legacy, index=False)

    print("\n" + "=" * 72)
    print("Daily stats written")
    print("=" * 72)
    print(f"Combined: {out_combined_new.name}")
    print(f"  rows: {len(combined_daily):,}")
    print(f"  files (rel_path): {combined_daily['rel_path'].nunique():,}")
    print(f"  datasets: {sorted([str(x) for x in combined_daily['dataset'].cat.categories]) if hasattr(combined_daily['dataset'], 'cat') else combined_daily['dataset'].unique().tolist()}")
    print(f"Legacy copy: {out_combined_legacy.name} (for backwards compatibility)")

    # Optional: per dataset files
    if args.write_per_dataset:
        for ds, ddf in daily_ds_frames.items():
            out_ds = data_dir / daily_out_name(ds)
            ddf.to_parquet(out_ds, index=False)
            print(f"Per-dataset: {out_ds.name}  | rows={len(ddf):,} | files={ddf['rel_path'].nunique():,}")

    # Small summary by dataset/scenario
    try:
        summary = (
            combined_daily.groupby(["dataset", "scenario"], observed=True)
            .agg(rows=("DBT_max", "size"), files=("rel_path", "nunique"))
            .reset_index()
            .sort_values(["dataset", "scenario"])
        )
        print("\nDataset/scenario summary (rows / files):")
        print(summary.to_string(index=False))
    except Exception:
        pass

    # Compute derived aggregations from HOURLY data (percentiles from hourly = more robust)
    if not args.skip_derived:
        idx = load_index_from_dir(data_dir)
        if not idx.empty:
            combined_hourly = pd.concat(hourly_parts, axis=0, ignore_index=False)
            # Ensure rel_path is normalized for merge with index
            if "rel_path" in combined_hourly.columns:
                combined_hourly["rel_path"] = combined_hourly["rel_path"].astype(str).str.replace("\\", "/", regex=False)
            compute_derived_stats_from_hourly(
                hourly=combined_hourly,
                idx=idx,
                data_dir=data_dir,
                percentiles=args.percentiles,
                verbose=args.verbose,
            )
        else:
            print("\n⚠️  No index found - skipping derived aggregations")
            print("   Run script 05 to create D-*__epw_index.json files first")
    
    print("\n" + "=" * 72)
    print("✅ All done!")
    print("=" * 72)
    print("\nNext steps:")
    print("1) Clear Streamlit cache: streamlit cache clear")
    print("2) Restart the app: streamlit run app.py")
    print("\nExpected speedup: ~13 seconds on first load (when cache is cold)")
    print()


if __name__ == "__main__":
    main()
