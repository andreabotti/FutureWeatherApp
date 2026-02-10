#!/usr/bin/env python3
"""
06D_precompute_daily_profiles.py

Precomputes daily DBT profiles (f22e single-variant and f22d baseline-vs-compare) from
the same daily stats and index used by the app, so the app can load them for D3 charts
instead of calling f22e/f22d at runtime.

Requires 06B to have run first (daily stats and inventory in _tables).

Outputs under: <parquet-root>/_tables/

- D-TMYxFWG__DailyProfilesAbs__<variant>__<max|mean>.parquet
    Long format: location_id, location_name, day_of_year, DBT (one row per location per day).

- D-TMYxFWG__DailyProfilesDelta__<baseline>__<compare_label>__<max|mean>.parquet
    Long format: location_id, location_name, role (base|comp), day_of_year, DBT.
    When compare is baseline__suffix (e.g. tmyx_2009-2023__rcp45_2080), compare_label
    is the suffix only (rcp45_2080) so the baseline is not repeated in the filename.

Variant names in filenames use double underscore; internal variant strings stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


BASELINE_VARIANTS = ["tmyx", "tmyx_2004-2018", "tmyx_2007-2021", "tmyx_2009-2023"]
FUTURE_RE = re.compile(r"^(?P<base>tmyx(?:_\d{4}-\d{4})?)__rcp(?P<rcp>\d{2})_(?P<year>\d{4})$", re.IGNORECASE)


def _parse_inventory_cols(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [c.strip() for c in str(value).split(";") if c.strip()]


def _parse_station_key(station_key: str) -> tuple[str, str]:
    key = str(station_key or "").strip()
    m = re.match(r"^(.*)\.(\d{6})$", key)
    if m:
        return m.group(1), m.group(2)
    return key, key


def _normalize_daily_stats_b(daily_stats: pd.DataFrame) -> pd.DataFrame:
    if daily_stats is None or daily_stats.empty:
        return pd.DataFrame()
    df = daily_stats.copy()
    if "station_key" in df.columns:
        df["station_key"] = df["station_key"].astype(str).str.strip()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["month"] = df["date"].dt.month.astype(int)
        df["day"] = df["date"].dt.day.astype(int)
    if "Tmax" in df.columns and "DBT_max" not in df.columns:
        df = df.rename(columns={"Tmax": "DBT_max"})
    if "Tmean" in df.columns and "DBT_mean" not in df.columns:
        df = df.rename(columns={"Tmean": "DBT_mean"})
    if "station_key" in df.columns and "scenario" in df.columns:
        df["rel_path"] = df["station_key"].astype(str) + "__" + df["scenario"].astype(str)
    return df


def _build_idx_from_inventory(inventory: pd.DataFrame, epw_meta: dict | None = None) -> pd.DataFrame:
    if inventory is None or inventory.empty:
        return pd.DataFrame()
    inv = inventory.copy()
    inv["station_key"] = inv["station_key"].astype(str).str.strip()
    name_col = "station_name" if "station_name" in inv.columns else "location_name" if "location_name" in inv.columns else None
    lat_col = "latitude" if "latitude" in inv.columns else "lat" if "lat" in inv.columns else None
    lon_col = "longitude" if "longitude" in inv.columns else "lon" if "lon" in inv.columns else None
    epw_meta = epw_meta or {}
    rows = []
    for _, r in inv.iterrows():
        station_key = str(r["station_key"])
        region = r.get("region")
        scenarios = _parse_inventory_cols(r.get("cols"))
        if not scenarios:
            continue
        loc_name, station_id = _parse_station_key(station_key)
        for scenario in scenarios:
            scenario = str(scenario).strip()
            meta = epw_meta.get(str(station_id))
            rows.append({
                "station_key": station_key,
                "location_id": station_id,
                "location_name": r.get(name_col) if name_col else loc_name,
                "latitude": r.get(lat_col) if lat_col else (meta or {}).get("latitude", pd.NA),
                "longitude": r.get(lon_col) if lon_col else (meta or {}).get("longitude", pd.NA),
                "region": region,
                "variant": scenario,
                "group_key": station_key,
                "rel_path": f"{station_key}__{scenario}",
                "is_cti": False,
            })
    return pd.DataFrame(rows)


def _load_epw_meta_optional(fwg_root: Path) -> dict:
    path = fwg_root / "data" / "03__italy_all_epw_DBT_streamlit" / "D-TMY__epw_index.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return {}
    meta = {}
    for r in records:
        station_id = str(r.get("station_id") or r.get("wmo") or "").strip()
        if not station_id:
            continue
        meta[station_id] = {"latitude": r.get("latitude"), "longitude": r.get("longitude")}
    return meta


def _safe_filename(s: str) -> str:
    """Replace characters that are problematic in filenames."""
    return s.replace("/", "_").replace("\\", "_").replace(":", "_")


def _bundle_abs_to_long(bundle: dict, variant: str, daily_stat: str) -> pd.DataFrame:
    """Convert f22e bundle to long-format DataFrame."""
    keys = bundle.get("keys") or []
    profiles = bundle.get("profiles") or {}
    if not keys or not profiles:
        return pd.DataFrame()
    rows = []
    for loc, data in profiles.items():
        name = (data.get("name") or loc) if isinstance(data, dict) else loc
        series = data.get("series") if isinstance(data, dict) else []
        if not series:
            continue
        for day_idx, (m, d) in enumerate(keys):
            if day_idx >= len(series):
                break
            val = series[day_idx]
            rows.append({
                "location_id": str(loc),
                "location_name": str(name),
                "variant": variant,
                "daily_stat": daily_stat,
                "day_of_year": day_idx + 1,
                "DBT": float(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else None,
            })
    return pd.DataFrame(rows)


def _bundle_delta_to_long(bundle: dict, baseline_variant: str, compare_variant: str, daily_stat: str) -> pd.DataFrame:
    """Convert f22d bundle to long-format DataFrame."""
    keys = bundle.get("keys") or []
    profiles = bundle.get("profiles") or {}
    if not keys or not profiles:
        return pd.DataFrame()
    rows = []
    for loc, data in profiles.items():
        name = (data.get("name") or loc) if isinstance(data, dict) else loc
        for role in ("base", "comp"):
            arr = data.get(role) if isinstance(data, dict) else []
            if not arr:
                continue
            for day_idx, (m, d) in enumerate(keys):
                if day_idx >= len(arr):
                    break
                val = arr[day_idx]
                rows.append({
                    "location_id": str(loc),
                    "location_name": str(name),
                    "baseline_variant": baseline_variant,
                    "compare_variant": compare_variant,
                    "daily_stat": daily_stat,
                    "role": role,
                    "day_of_year": day_idx + 1,
                    "DBT": float(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else None,
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute daily profiles for D3 charts (f22e/f22d).")
    ap.add_argument("--parquet-root", required=True, help="data/04__italy_tmy_fwg_parquet (must have _tables from 06B)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.parquet_root).resolve()
    tables = root / "_tables"
    daily_path = tables / "D-TMYxFWG__DBT__F-DD__L-ALL.parquet"
    inv_path = tables / "D-TMYxFWG__Inventory__F-NA__L-ALL.parquet"

    if not daily_path.exists() or not inv_path.exists():
        print(f"Missing {daily_path.name} or {inv_path.name}. Run 06B first.")
        return 1

    script_path = Path(__file__).resolve()
    fwg_root = script_path.parent.parent.parent
    sys.path.insert(0, str(fwg_root))
    try:
        import libs.fn__libs as h
    except ImportError as e:
        print(f"Failed to import libs.fn__libs: {e}")
        return 1

    daily_stats = pd.read_parquet(daily_path)
    inventory = pd.read_parquet(inv_path)
    daily_norm = _normalize_daily_stats_b(daily_stats)
    if daily_norm.empty:
        print("Normalized daily stats empty.")
        return 1

    epw_meta = _load_epw_meta_optional(fwg_root)
    idx = _build_idx_from_inventory(inventory, epw_meta)
    if idx.empty:
        print("Index empty.")
        return 1

    all_location_ids = tuple(idx["location_id"].dropna().unique().astype(str).tolist())
    if not all_location_ids:
        print("No location_ids in index.")
        return 1

    variants = sorted(idx["variant"].dropna().unique().astype(str).tolist())
    variants = [v for v in variants if "cti" not in v.lower()]
    baselines = [v for v in variants if v in BASELINE_VARIANTS or (v.startswith("tmyx") and "__" not in v)]
    if not baselines:
        baselines = [variants[0]] if variants else []
    compare_variants = [v for v in variants if FUTURE_RE.match(v) or "rcp" in v.lower()]
    if not compare_variants and len(variants) > 1:
        compare_variants = variants[1:]

    first_baseline = baselines[0] if baselines else None
    written = 0

    print("--- Daily profiles (single variant, for D3 absolute charts) ---")
    for variant in variants:
        for daily_stat in ("max", "mean"):
            out_name = f"D-TMYxFWG__DailyProfilesAbs__{_safe_filename(variant)}__{daily_stat}.parquet"
            out_path = tables / out_name
            if out_path.exists() and not args.overwrite:
                if args.verbose:
                    print(f"  SKIP {out_name}")
                continue
            try:
                bundle = h.f22e__build_daily_db_profiles_single_variant_from_daily_stats(
                    daily_norm,
                    idx,
                    location_ids=all_location_ids,
                    variant=variant,
                    daily_stat=daily_stat,
                    baseline_variant=first_baseline,
                )
                df = _bundle_abs_to_long(bundle, variant, daily_stat)
                if not df.empty:
                    df.to_parquet(out_path, index=False)
                    print(f"  Wrote {out_name} ({len(df):,} rows)")
                    written += 1
            except Exception as e:
                print(f"  {out_name}: {e}")

    print("--- Daily profiles (delta baseline vs compare, for D3 delta charts) ---")
    for baseline in baselines:
        for compare in compare_variants:
            if baseline == compare:
                continue
            # When compare is "baseline__rcp45_2080", use suffix only so baseline isn't repeated in filename
            compare_label = compare[len(baseline) + 2:] if compare.startswith(baseline + "__") else compare
            for daily_stat in ("max", "mean"):
                out_name = f"D-TMYxFWG__DailyProfilesDelta__{_safe_filename(baseline)}__{_safe_filename(compare_label)}__{daily_stat}.parquet"
                out_path = tables / out_name
                if out_path.exists() and not args.overwrite:
                    if args.verbose:
                        print(f"  SKIP {out_name}")
                    continue
                try:
                    bundle = h.f22d__build_daily_db_profiles_from_daily_stats(
                        daily_norm,
                        idx,
                        location_ids=all_location_ids,
                        baseline_variant=baseline,
                        compare_variant=compare,
                        daily_stat=daily_stat,
                    )
                    df = _bundle_delta_to_long(bundle, baseline, compare, daily_stat)
                    if not df.empty:
                        df.to_parquet(out_path, index=False)
                        print(f"  Wrote {out_name} ({len(df):,} rows)")
                        written += 1
                except Exception as e:
                    print(f"  {out_name}: {e}")

    print("============================================================")
    print(f"06D complete. Wrote {written} profile table(s).")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
