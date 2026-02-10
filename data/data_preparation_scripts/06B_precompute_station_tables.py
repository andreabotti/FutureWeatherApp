#!/usr/bin/env python3
"""
06B_precompute_station_tables.py

Reads per-station parquets written by 05B and writes app-ready tables:

Outputs under: <parquet-root>/_tables/

- D-TMYxFWG__DBT__F-DD__L-ALL.parquet
    Tidy daily stats: region, station_key, scenario, date, Tmax, Tmean, Tmin

- D-TMYxFWG__Inventory__F-NA__L-ALL.parquet
    Inventory per station: region, station_key, n_cols, cols (semicolon), and counts by type

- pairing_debug.csv
    For each station_key + baseline_variant: which future columns exist/missing

- D-TMYxFWG__FileStats__P-{95,97,99}.parquet
    Per-file Tmax percentile and Tavg mean (one file per percentile).

- D-TMYxFWG__LocationStats__P-{95,97,99}.parquet
    Per-location absolute stats per variant (one file per percentile).

- D-TMYxFWG__LocationDeltas__P-{95,97,99}.parquet
    Per-location deltas for each baseline/compare pair (one file per percentile).

- D-TMYxFWG__MonthlyDeltas__F-MM__P-{95,97,99}.parquet
    Monthly delta table for each baseline/compare/metric (one file per percentile).
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

EXPECTED_RCP_TAGS = ["rcp26_2050", "rcp26_2080", "rcp45_2050", "rcp45_2080", "rcp85_2050", "rcp85_2080"]

# Output names (adopted naming scheme)
DAILY_OUT_NAME = "D-TMYxFWG__DBT__F-DD__L-ALL.parquet"
INVENTORY_OUT_NAME = "D-TMYxFWG__Inventory__F-NA__L-ALL.parquet"
PAIRING_DEBUG_NAME = "pairing_debug.csv"
PERCENTILES = [95, 97, 99]


def discover_station_parquets(root: Path):
    # root/<REGION>/<station_key>.parquet (excluding _tables)
    for region_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "_tables"]):
        region = region_dir.name
        for p in sorted(region_dir.glob("*.parquet")):
            yield region, p


def _parse_inventory_cols(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [c.strip() for c in str(value).split(";") if c.strip()]


def _parse_station_key(station_key: str) -> tuple[str, str]:
    """Return (location_name, station_id) from station_key."""
    key = str(station_key or "").strip()
    m = re.match(r"^(.*)\.(\d{6})$", key)
    if m:
        return m.group(1), m.group(2)
    return key, key


def _normalize_daily_stats_b(daily_stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize B-route daily stats to DBT_max, DBT_mean, rel_path, month, day for libs."""
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


def _build_idx_from_inventory(
    inventory: pd.DataFrame,
    epw_meta: dict | None = None,
) -> pd.DataFrame:
    """Build index DataFrame compatible with app's _build_idx_from_inventory (no streamlit)."""
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
    """Load epw index meta from 03 dir if present (for lat/lon)."""
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
        meta[station_id] = {
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
        }
    return meta


def run_derived_precompute(
    tables: Path,
    daily_stats: pd.DataFrame,
    inventory: pd.DataFrame,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Compute and write FileStats, LocationStats, LocationDeltas, MonthlyDeltas for percentiles 95, 97, 99."""
    script_path = Path(__file__).resolve()
    fwg_root = script_path.parent.parent.parent
    libs_path = fwg_root / "libs"
    if not libs_path.exists():
        if verbose:
            print("Skip derived stats: libs not found")
        return
    sys.path.insert(0, str(fwg_root))
    try:
        import libs.fn__libs as h
    except ImportError as e:
        if verbose:
            print(f"Skip derived stats: {e}")
        return
    daily_norm = _normalize_daily_stats_b(daily_stats)
    if daily_norm.empty:
        if verbose:
            print("Skip derived stats: normalized daily stats empty")
        return
    epw_meta = _load_epw_meta_optional(fwg_root)
    idx = _build_idx_from_inventory(inventory, epw_meta)
    if idx.empty:
        if verbose:
            print("Skip derived stats: index empty")
        return
    variants = sorted(idx["variant"].dropna().unique().astype(str).tolist())
    variants = [v for v in variants if "cti" not in v.lower()]
    if not variants:
        if verbose:
            print("Skip derived stats: no variants")
        return
    baselines = [v for v in variants if v in BASELINE_VARIANTS or (v.startswith("tmyx") and "__" not in v)]
    if not baselines:
        baselines = [variants[0]] if variants else []
    compare_variants = [v for v in variants if FUTURE_RE.match(v) or "rcp" in v.lower() or "ssp" in v.lower()]
    if not compare_variants and len(variants) > 1:
        compare_variants = variants[1:]
    print("\n--- Derived stats (percentiles 95, 97, 99) ---")
    for pct in PERCENTILES:
        p_dec = pct / 100.0
        # FileStats
        out_fs = tables / f"D-TMYxFWG__FileStats__P-{pct}.parquet"
        if out_fs.exists() and not overwrite:
            if verbose:
                print(f"  SKIP FileStats P-{pct}")
        else:
            try:
                fs = h.f123__build_file_stats_from_daily(daily_norm, p_dec)
                if not fs.empty:
                    fs.to_parquet(out_fs, index=False)
                    print(f"  Wrote {out_fs.name} ({len(fs):,} rows)")
            except Exception as e:
                print(f"  FileStats P-{pct}: {e}")
        # LocationStats
        out_ls = tables / f"D-TMYxFWG__LocationStats__P-{pct}.parquet"
        if out_ls.exists() and not overwrite:
            if verbose:
                print(f"  SKIP LocationStats P-{pct}")
        else:
            parts = []
            for v in variants:
                try:
                    st_df = h.f28b__compute_location_stats_for_variant_from_daily(
                        daily_norm, idx, variant=v, percentile=p_dec
                    )
                    if not st_df.empty:
                        st_df["variant"] = v
                        st_df["percentile"] = p_dec
                        parts.append(st_df)
                except Exception as e:
                    if verbose:
                        print(f"    {v}: {e}")
            if parts:
                out_df = pd.concat(parts, ignore_index=True)
                out_df.to_parquet(out_ls, index=False)
                print(f"  Wrote {out_ls.name} ({len(out_df):,} rows, {len(variants)} variants)")
        # LocationDeltas
        out_ld = tables / f"D-TMYxFWG__LocationDeltas__P-{pct}.parquet"
        if out_ld.exists() and not overwrite:
            if verbose:
                print(f"  SKIP LocationDeltas P-{pct}")
        else:
            parts = []
            for b in baselines:
                for c in compare_variants:
                    if b == c:
                        continue
                    try:
                        delta_df = h.f125__compute_location_deltas_from_daily(
                            daily_norm, idx,
                            baseline_variant=b, compare_variant=c, percentile=p_dec,
                        )
                        if not delta_df.empty:
                            delta_df["baseline_variant"] = b
                            delta_df["compare_variant"] = c
                            delta_df["percentile"] = p_dec
                            parts.append(delta_df)
                    except Exception as e:
                        if verbose:
                            print(f"    {b} vs {c}: {e}")
            if parts:
                out_df = pd.concat(parts, ignore_index=True)
                out_df.to_parquet(out_ld, index=False)
                print(f"  Wrote {out_ld.name} ({len(out_df):,} rows)")
        # MonthlyDeltas
        out_md = tables / f"D-TMYxFWG__MonthlyDeltas__F-MM__P-{pct}.parquet"
        if out_md.exists() and not overwrite:
            if verbose:
                print(f"  SKIP MonthlyDeltas P-{pct}")
        else:
            parts = []
            for b in baselines:
                for c in compare_variants:
                    if b == c:
                        continue
                    for metric_key in ["dTmax", "dTavg"]:
                        try:
                            md = h.f124__build_monthly_delta_table(
                                daily_norm, idx,
                                baseline_variant=b, compare_variant=c,
                                percentile=p_dec, metric_key=metric_key,
                            )
                            if not md.empty:
                                md["baseline_variant"] = b
                                md["compare_variant"] = c
                                md["metric_key"] = metric_key
                                md["percentile"] = p_dec
                                parts.append(md)
                        except Exception as e:
                            if verbose:
                                print(f"    {b} vs {c} {metric_key}: {e}")
            if parts:
                out_df = pd.concat(parts, ignore_index=True)
                out_df.to_parquet(out_md, index=False)
                print(f"  Wrote {out_md.name} ({len(out_df):,} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-root", required=True, help="data/04__italy_tmy_fwg_parquet")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.parquet_root).resolve()
    tables = root / "_tables"
    tables.mkdir(parents=True, exist_ok=True)

    daily_rows = []
    pairing_rows = []
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

        # ---- Inventory
        n_baselines = sum(1 for c in cols_str if c in BASELINE_VARIANTS)
        n_futures = sum(1 for c in cols_str if FUTURE_RE.match(c))
        n_other = len(cols_str) - n_baselines - n_futures

        inventory_rows.append(
            {
                "region": region,
                "station_key": station_key,
                "rows": int(len(df)),
                "n_cols": int(len(cols_str)),
                "n_baselines": int(n_baselines),
                "n_futures": int(n_futures),
                "n_other": int(n_other),
                "cols": ";".join(cols_str),
            }
        )

        # ---- Pairing debug (baseline variant -> expected futures)
        for base in BASELINE_VARIANTS:
            missing = []
            future_present = 0

            if base not in cols_str:
                missing.append(base)

            for tag in EXPECTED_RCP_TAGS:
                fut = f"{base}__{tag}"
                if fut not in cols_str:
                    missing.append(fut)
                else:
                    future_present += 1

            pairing_rows.append(
                {
                    "region": region,
                    "station_key": station_key,
                    "baseline_variant": base,
                    "future_present_count": future_present,
                    "missing_count": len(missing),
                    "missing_items": ";".join(missing),
                }
            )

        # ---- Daily stats for all scenario columns
        for scenario in cols_str:
            s = df[scenario].dropna()
            if s.empty:
                continue
            g = s.resample("D")
            out = pd.DataFrame(
                {
                    "Tmax": g.max(),
                    "Tmean": g.mean(),
                    "Tmin": g.min(),
                }
            ).reset_index().rename(columns={"dt": "date"})
            out["region"] = region
            out["station_key"] = station_key
            out["scenario"] = scenario
            daily_rows.append(out)

        if args.verbose:
            print(f"[{region}] {station_key}: cols={len(cols_str)} rows={len(df)}")

    daily_stats = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    pairing_debug = pd.DataFrame(pairing_rows)
    inventory = pd.DataFrame(inventory_rows)

    daily_path = tables / DAILY_OUT_NAME
    inv_path = tables / INVENTORY_OUT_NAME
    pair_path = tables / PAIRING_DEBUG_NAME

    # Write daily stats parquet
    if daily_path.exists() and not args.overwrite:
        print(f"SKIP daily exists: {daily_path}")
    else:
        tmp = daily_path.with_suffix(".parquet.tmp")
        daily_stats.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(daily_path)

    # Write inventory parquet
    if inv_path.exists() and not args.overwrite:
        print(f"SKIP inventory exists: {inv_path}")
    else:
        tmp = inv_path.with_suffix(".parquet.tmp")
        inventory.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(inv_path)

    # Pairing debug CSV
    pairing_debug.to_csv(pair_path, index=False)

    # Derived percentile-based stats (FileStats, LocationStats, LocationDeltas, MonthlyDeltas)
    run_derived_precompute(tables, daily_stats, inventory, overwrite=args.overwrite, verbose=args.verbose)

    print("============================================================")
    print("06B complete")
    print(f"daily_stats rows : {len(daily_stats):,}")
    print(f"inventory rows   : {len(inventory):,}")
    print(f"pairing rows     : {len(pairing_debug):,}")
    print(f"Wrote: {daily_path}")
    print(f"Wrote: {inv_path}")
    print(f"Wrote: {pair_path}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
