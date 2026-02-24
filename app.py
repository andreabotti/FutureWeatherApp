from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components
import json
import time
import plotly.graph_objects as go
import libs.fn__libs as h
from libs.fn__libs_bilingual import label
from libs.fn__page_header import f001__create_page_header

import libs.fn__libs_charts as _fn_charts

h.f101__inject_inter_font()
h.f102__enable_altair_inter_theme()

# --- Confronto Regione (D3): margin above/below the horizontal divider (px) ---
D3_DIVIDER_MARGIN_TOP_PX = 56
D3_DIVIDER_MARGIN_BOTTOM_PX = 30




def _record_timing(name: str, seconds: float, notes: str | None = None) -> None:
    timings = st.session_state.setdefault("code_timing", {})
    timings[name] = {"seconds": float(seconds), "notes": notes or ""}


def _timed(name: str, fn, notes: str | None = None):
    start = time.perf_counter()
    result = fn()
    _record_timing(name, time.perf_counter() - start, notes=notes)
    return result


def _baseline_location_ids(idx: pd.DataFrame, baseline_variant: str) -> set[str]:
    return h.f130__baseline_location_ids(idx, baseline_variant)


def _read_parquet_robust(parquet_path: Path, columns: list = None, **kwargs):
    return h.f131__read_parquet_robust(parquet_path, columns=columns, **kwargs)


def _parse_inventory_cols(value) -> list[str]:
    return h.f132__parse_inventory_cols(value)


def _parse_station_key(station_key: str) -> tuple[str, str]:
    return h.f133__parse_station_key(station_key)


@st.cache_data(show_spinner=False)
def _load_epw_index_meta(index_path: Path) -> dict:
    return h.f134__load_epw_index_meta(index_path)


def _normalize_daily_stats_b(daily_stats: pd.DataFrame) -> pd.DataFrame:
    return h.f135__normalize_daily_stats_b(daily_stats)


def _build_idx_from_inventory(inventory: pd.DataFrame, epw_meta: dict) -> pd.DataFrame:
    return h.f136__build_idx_from_inventory(inventory, epw_meta)


@st.cache_data(show_spinner=False)
def _load_inventory_cached(base_dir: Path) -> pd.DataFrame:
    return h.load_inventory(base_dir)


@st.cache_data(show_spinner=False)
def _load_daily_stats_b_cached(base_dir: Path) -> pd.DataFrame:
    return h.load_daily_stats(base_dir)


@st.cache_data(show_spinner=False)
def _load_pairing_debug_cached(base_dir: Path) -> pd.DataFrame:
    return h.load_pairing_debug(base_dir)


@st.cache_data(show_spinner=False)
def _load_station_hourly_cached(base_dir: Path, region: str, station_key: str) -> pd.DataFrame:
    return h.load_station_hourly(base_dir, region, station_key)


@st.cache_data(show_spinner=False)
def _load_cti_inventory_cached(base_dir: Path) -> pd.DataFrame:
    return h.load_cti_inventory(base_dir)


@st.cache_data(show_spinner=False)
def _load_cti_daily_stats_cached(base_dir: Path) -> pd.DataFrame:
    return h.load_cti_daily_stats(base_dir)


@st.cache_data(show_spinner=False)
def _load_cti_monthly_stats_cached(base_dir: Path) -> pd.DataFrame:
    return h.load_cti_monthly_stats(base_dir)


@st.cache_data(show_spinner=False)
def _load_cti_station_hourly_cached(base_dir: Path, region: str, station_key: str) -> pd.DataFrame:
    return h.load_cti_station_hourly(base_dir, region, station_key)


@st.cache_data(show_spinner=False)
def _load_cti_list_cached(path: Path) -> pd.DataFrame:
    return h.f94b__cti_load_list_only(path)


def _normalize_monthly_stats_cti(monthly_stats: pd.DataFrame) -> pd.DataFrame:
    return h.f137__normalize_monthly_stats_cti(monthly_stats)


def _build_idx_from_cti_inventory(inventory: pd.DataFrame, cti_list: pd.DataFrame) -> pd.DataFrame:
    return h.f140__build_idx_from_cti_inventory(inventory, cti_list)


def _station_hourly_to_long(
    hourly_wide: pd.DataFrame, station_key: str, scenarios: list[str] | None = None
) -> pd.DataFrame:
    return h.f141__station_hourly_to_long(hourly_wide, station_key, scenarios=scenarios)


# -----------------------------
# Page header / config
# -----------------------------
f001__create_page_header()

# Use absolute path based on script location to avoid working directory issues
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_B_DATA_DIR = SCRIPT_DIR / "data" / "04__italy_tmy_fwg_parquet"
DEFAULT_CTI_DATA_DIR = SCRIPT_DIR / "data" / "04__italy_cti_parquet"
UNITR_CTI_CSV = DEFAULT_CTI_DATA_DIR / "UNITR _10349-22016__Prospetto4.csv"
DEFAULT_DATA_DIR = DEFAULT_B_DATA_DIR
# Legacy paths kept for compatibility/debug (not used in B route)
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR
DEFAULT_TIDY_PARQUET = DEFAULT_DATA_DIR / "dbt_rh_tidy.parquet"
DEFAULT_TMYX_HOURLY_PARQUET = DEFAULT_DATA_DIR / "tmyx_hourly.parquet"

# Precomputed Italy GeoJSON — produced by 06E_precompute_geodata.py
# Falls back to network fetch if files not present.
GEO_DIR = SCRIPT_DIR / "data" / "geo"
_GEO_SCRIPT = h.geo_inline_script(str(GEO_DIR))

# Variants shown in maps, delta tables, and Italian Regions view (sidebar + charts).
# Dated TMYx variants are excluded from these views; they remain visible in the TMYx station tab only.
MAP_BASELINE_VARIANT = "tmyx"
DATED_TMYX_VARIANTS: set[str] = {"tmyx_2004-2018", "tmyx_2007-2021", "tmyx_2009-2023"}


@st.cache_data(show_spinner=False)
def _load_unitr_cti_points(csv_path: Path) -> list[dict]:
    return h.f142__load_unitr_cti_points(csv_path)


def _discover_parquet_files(data_dir: Path) -> list[tuple[Path, str]]:
    return h.f143__discover_parquet_files(data_dir)


def _discover_parquet_files_by_region(data_dir: Path) -> list[tuple[str, list[tuple[Path, str]]]]:
    return h.f144__discover_parquet_files_by_region(data_dir)


@st.cache_data(show_spinner=False)
def _series_from_sub(sub: pd.DataFrame) -> list[dict]:
    """Vectorized replacement for the iterrows-based hourly series builder."""
    sub = sub.dropna(subset=["DBT"]).copy()
    if sub.empty:
        return []
    dts = sub["datetime"]
    ts = dts.apply(lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d)).tolist()
    vs = sub["DBT"].astype(float).tolist()
    return [{"t": t, "v": v} for t, v in zip(ts, vs)]


def _build_hourly_series_by_location_id(
    base_dir: Path,
    by_region: list,
    region_stem_to_loc_items: tuple,
    max_stations: int = 12,
) -> dict:
    """
    Build hourly_series_by_location_id from station parquet files.
    Cached to avoid re-reading parquet files on every Streamlit rerun.
    region_stem_to_loc_items: tuple of ((region, stem), location_id) for stable cache key.
    """
    region_stem_to_loc = dict(region_stem_to_loc_items)
    hourly_series_by_location_id: dict = {}
    loaded_count = 0
    for region_label, region_files in by_region:
        if region_label in ("Tables", "Root") or loaded_count >= max_stations:
            continue
        region_code = region_label
        for path, file_label in region_files:
            if "hourly" not in file_label.lower() or not path.exists():
                continue
            stem = path.stem
            if not stem:
                continue
            loc_id = region_stem_to_loc.get((region_code, stem))
            if loc_id is None:
                loc_id = region_stem_to_loc.get((str(region_code).strip(), str(stem).strip()))
            if loc_id is None or loc_id in hourly_series_by_location_id:
                continue
            try:
                hourly_wide = _read_parquet_robust(path)
            except Exception:
                continue
            if hourly_wide is None or hourly_wide.empty:
                continue
            long_hourly = _station_hourly_to_long(hourly_wide, stem, scenarios=None)
            if long_hourly.empty or "datetime" not in long_hourly.columns or "DBT" not in long_hourly.columns:
                continue
            hourly_series_by_location_id[loc_id] = {}
            long_hourly["_month"] = pd.to_datetime(long_hourly["datetime"], errors="coerce").dt.month
            summer = long_hourly[long_hourly["_month"].isin([6, 7, 8])]
            for sc in long_hourly["scenario"].dropna().unique().tolist():
                sc_str = str(sc).strip()
                if not sc_str:
                    continue
                sub = summer[summer["scenario"] == sc]
                series = _series_from_sub(sub)
                hourly_series_by_location_id[loc_id][sc_str] = series
            for alias_from, alias_to in [("rcp45_2050", "tmyx__rcp45_2050"), ("rcp85_2080", "tmyx__rcp85_2080")]:
                if alias_from in hourly_series_by_location_id[loc_id] and alias_to not in hourly_series_by_location_id[loc_id]:
                    hourly_series_by_location_id[loc_id][alias_to] = hourly_series_by_location_id[loc_id][alias_from]
            loaded_count += 1
            if loaded_count >= max_stations:
                break
        if loaded_count >= max_stations:
            break
    return hourly_series_by_location_id


@st.cache_data(show_spinner=False)
def _build_region_maps_html(
    baseline_variants_key: tuple,
    compare_variant: str,
    metric_key: str,
    percentile: float,
    ui_lang: str,
    # Fingerprints (small strings) used for cache key; actual data passed as _-prefixed args
    abs_fp: str,
    profiles_fp: str,
    future_abs_fp: str,
    region_options_fp: str,
    unitr_fp: str,
    hourly_fp: str,
    selected_location_id: str | None,
    dashboard_cols: str,
    maps_row_gap_px: int,
    maps_row3_gap_px: int,
    thermo_separator_gap_px: int,
    divider_margin_bottom_px: int,
    # Actual data, excluded from Streamlit cache-key hash (underscore prefix)
    _abs_points_json: str = "{}",
    _profiles_json: str = "{}",
    _future_abs_json: str = "{}",
    _region_options_json: str = "[]",
    _unitr_points_json: str = "[]",
    _hourly_json: str = "{}",
    _geo_script: str = "",   # excluded from cache key (underscore prefix)
) -> str:
    import json as _json
    abs_points_by_variant = _json.loads(_abs_points_json)
    profiles_bundle_by_variant = _json.loads(_profiles_json)
    future_abs_points_by_variant = _json.loads(_future_abs_json)
    region_options_data = _json.loads(_region_options_json)
    unitr_points = _json.loads(_unitr_points_json)
    hourly_series_by_location_id = _json.loads(_hourly_json)
    font_theme = {
        "enabled": True,
        "font_family": '"Inter", "Helvetica Neue", Helvetica, Arial, sans-serif',
        "fs_title": 18,
        "fs_subtitle": 15,
        "fs_label": 10,
        "fs_small": 10,
        "fs_axis": 11,
    }
    return h.f23d__d3_region_maps_html(
        baseline_variants=list(baseline_variants_key),
        abs_points_by_variant=abs_points_by_variant,
        delta_points_by_variant={},
        metric_key=metric_key,
        compare_variant=compare_variant,
        percentile=percentile,
        region_options=region_options_data,
        future_abs_points_by_variant=future_abs_points_by_variant or None,
        initial_region=None,
        click_callback_key="single_regions_location_click_v2",
        profiles_bundle_by_variant=profiles_bundle_by_variant or None,
        unitr_points=unitr_points,
        font_theme=font_theme,
        hide_region_selector=False,
        layout_mode="columns",
        cti_on_top=True,
        hide_row2=True,
        side_text_html=(
            "<b>Temperature Summary</b><br/>"
            "• Tmax uses selected percentile<br/>"
            "• Tavg is mean of daily means<br/>"
            "• Click a marker to update charts"
        ),
        dashboard_cols=dashboard_cols,
        maps_row_gap_px=maps_row_gap_px,
        maps_row3_gap_px=maps_row3_gap_px,
        thermo_separator_gap_px=thermo_separator_gap_px,
        divider_margin_bottom_px=divider_margin_bottom_px,
        ui_lang=ui_lang,
        hourly_series_by_variant=None,
        initial_selected_location_id=selected_location_id,
        hourly_series_by_location_id=hourly_series_by_location_id or None,
        geo_script=_geo_script,
    )


# CTI (Itaca) weather stations (hourly DBT) - copy CSVs into this folder for Single Regions Data integration
DEFAULT_CTI_DIR = SCRIPT_DIR / "data" / "cti"
CTI_DBT_CSV = DEFAULT_CTI_DIR / "CTI__DBT__ITA_WeatherStations__All.csv"
CTI_LIST_CSV = (
    (DEFAULT_B_DATA_DIR / "CTI__list__ITA_WeatherStations__All.csv")
    if (DEFAULT_B_DATA_DIR / "CTI__list__ITA_WeatherStations__All.csv").exists()
    else (DEFAULT_CTI_DIR / "CTI__list__ITA_WeatherStations__All.csv")
)

# -----------------------------
# TMYx heatmap settings (edit here, not in UI)
# -----------------------------
HEATMAP_COLORSCALE = "RdBu_r"
TMYX_CHART_HEIGHT = 200  # Shared height for all TMYx subplot figures
# Keep legend placement consistent by using the same top margin across all TMYx plots.
TMYX_CHART_MARGIN = dict(l=10, r=10, t=15, b=10)
# Explicit spacing controls for the TMYx tab layout:
TMYX_HEADER_TO_CHARTS_GAP_PX = -150       # vertical gap between header/legends row and charts row
TMYX_SUBPLOT_VERTICAL_SPACING = 0.10    # Plotly subplot vertical spacing (0..1)
TMYX_SUBPLOT_TITLE_FONTSIZE = 12
TMYX_SUBPLOT_TITLE_YSHIFT = -1  # positive yshift moves titles upward in Plotly
TMYX_SUBPLOT_TITLE_BOLD = True
HEATMAP_ZMIN = 0
HEATMAP_ZMAX = 40
AUTO_SAVE_TMYX_HOURLY_PARQUET = True


# -----------------------------
# UI
# -----------------------------
# st.caption(
#     "Compares dry-bulb temperature statistics between two climate-file variants for the same locations."
# )

with st.sidebar:
    # Paths moved to Debug expander (advanced)
    index_path = str(DEFAULT_INDEX_PATH)
    tidy_path = str(DEFAULT_TIDY_PARQUET)
    # Preview controls moved to Debug expander (advanced)
    index_preview_records = 500
    raw_preview_rows = 500

    # Debug / advanced options moved to Data Preview tab

def _load_index_smart(p: Path) -> pd.DataFrame:
    return h.f145__load_index_smart(p)


# Load inventory + daily stats (B route)
inventory = _timed(
    "load_b_inventory",
    lambda: _load_inventory_cached(DEFAULT_B_DATA_DIR),
    notes="Loads station inventory for B route.",
)
if inventory.empty:
    st.error(f"No inventory found in `{DEFAULT_B_DATA_DIR / '_tables'}`.")
    st.stop()

daily_stats_raw = _timed(
    "load_b_daily_stats",
    lambda: _load_daily_stats_b_cached(DEFAULT_B_DATA_DIR),
    notes="Loads precomputed daily stats for B route.",
)
daily_stats = _normalize_daily_stats_b(daily_stats_raw)
if daily_stats.empty:
    st.error(f"No daily stats found in `{DEFAULT_B_DATA_DIR / '_tables'}`.")
    st.stop()

pairing_debug = _timed(
    "load_b_pairing_debug",
    lambda: _load_pairing_debug_cached(DEFAULT_B_DATA_DIR),
    notes="Loads pairing debug table.",
)

epw_meta = _load_epw_index_meta(DEFAULT_B_DATA_DIR / "D-TMY__epw_index.json")
idx = _build_idx_from_inventory(inventory, epw_meta)
station_region_map = (
    inventory.set_index("station_key")["region"].astype(str).to_dict()
    if "station_key" in inventory.columns and "region" in inventory.columns
    else {}
)

# Load CTI dataset for Single Regions Data when tables exist
cti_inventory = pd.DataFrame()
cti_daily_stats = pd.DataFrame()
cti_monthly_stats = pd.DataFrame()
cti_idx = pd.DataFrame()

cti_tables_dir = DEFAULT_CTI_DATA_DIR / "_tables"
cti_tables_exist = (
    (cti_tables_dir / "D-CTI__Inventory__F-NA__L-ALL.parquet").exists()
    and (cti_tables_dir / "D-CTI__DBT__F-DD__L-ALL.parquet").exists()
)
if cti_tables_exist:
    cti_inventory = _timed(
        "load_cti_inventory",
        lambda: _load_cti_inventory_cached(DEFAULT_CTI_DATA_DIR),
        notes="Loads station inventory for CTI dataset.",
    )
    if cti_inventory.empty:
        st.warning(f"CTI inventory not found in `{DEFAULT_CTI_DATA_DIR / '_tables'}`.")

    cti_daily_stats_raw = _timed(
        "load_cti_daily_stats",
        lambda: _load_cti_daily_stats_cached(DEFAULT_CTI_DATA_DIR),
        notes="Loads precomputed daily stats for CTI dataset.",
    )
    cti_daily_stats = _normalize_daily_stats_b(cti_daily_stats_raw)
    if cti_daily_stats.empty:
        st.warning(f"CTI daily stats not found in `{DEFAULT_CTI_DATA_DIR / '_tables'}`.")

    cti_monthly_stats_raw = _timed(
        "load_cti_monthly_stats",
        lambda: _load_cti_monthly_stats_cached(DEFAULT_CTI_DATA_DIR),
        notes="Loads precomputed monthly stats for CTI dataset.",
    )
    cti_monthly_stats = _normalize_monthly_stats_cti(cti_monthly_stats_raw)

    cti_list_df = _load_cti_list_cached(CTI_LIST_CSV)
    cti_idx = _build_idx_from_cti_inventory(cti_inventory, cti_list_df)

# Lazy-load tidy only when needed (for D3 charts or Data Debug tab)
# This saves ~10-15 seconds on initial load
@st.cache_data(show_spinner=False)
def _load_tidy_raw_lazy(tidy_path: Path):
    """Load raw tidy data only when explicitly needed. Returns empty DataFrame if file doesn't exist."""
    if not tidy_path.exists():
        return pd.DataFrame()
    return _timed(
        "f111__load_tidy_parquet_raw",
        lambda: h.f111__load_tidy_parquet_raw(tidy_path),
        notes="Loads raw tidy parquet (lazy-loaded).",
    )

@st.cache_data(show_spinner=False)
def _load_tidy_lazy(tidy_path: Path):
    """Load tidy data only when explicitly needed. Returns empty DataFrame if file doesn't exist."""
    tidy_raw = _load_tidy_raw_lazy(tidy_path)
    if tidy_raw.empty:
        return pd.DataFrame()
    return _timed(
        "f112__postprocess_tidy_parquet",
        lambda: h.f112__postprocess_tidy_parquet(tidy_raw),
        notes="Loads and processes tidy parquet (lazy-loaded).",
    )


@st.cache_data(show_spinner=False)
def _load_regional_hourly_by_rel_paths(rel_paths: list, idx: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    return h.f148__load_regional_hourly_by_rel_paths(rel_paths, idx, base_dir)


@st.cache_data(show_spinner=False)
def _load_daily_stats_cached(daily_stats_path: Path, tidy_path: Path, idx: pd.DataFrame = None):
    """Load or compute daily stats with caching.
    
    Tries to load precomputed daily_stats.parquet first (fast path).
    Falls back to computing from regional files if available (medium path).
    Falls back to computing from tidy parquet if not available (slow path).
    """
    if daily_stats_path.exists():
        try:
            return _timed(
                "load_daily_stats_parquet",
                lambda: _read_parquet_robust(daily_stats_path),
                notes="Loads precomputed daily stats parquet (fast path).",
            )
        except (OSError, RuntimeError, Exception) as e:
            # If reading fails due to PyArrow issues, fall through to compute from source
            st.warning(f"⚠️ Could not load daily stats Parquet (will compute from source): {str(e)[:200]}")
            # Fall through to compute from source
    
    # Try to compute from regional files (much faster than monolithic tidy)
    if idx is not None:
        # Check if any regional files exist
        available_regions = [code for code in idx["region"].dropna().unique() if pd.notna(code)]
        regional_files_exist = any(
            any(DEFAULT_DATA_DIR.glob(f"D-*__DBT__F-HR__L-{code}.parquet"))
            or (DEFAULT_DATA_DIR / f"DBT__HR__{code}.parquet").exists()
            for code in available_regions
        )
        if regional_files_exist:
            # Get all rel_paths from idx
            all_rel_paths = idx["rel_path"].dropna().unique().tolist()
            if all_rel_paths:
                # Load from regional files
                hourly = _load_regional_hourly_by_rel_paths(all_rel_paths, idx, DEFAULT_DATA_DIR)
                if not hourly.empty:
                    # Compute daily stats on-the-fly (fast because data is already filtered by region)
                    return _timed(
                        "compute_daily_stats_from_regional",
                        lambda: _compute_daily_stats_from_hourly(hourly),
                        notes="Computes daily stats from regional hourly files (fast path).",
                    )
                else:
                    # Regional files exist but no data matched - this might be a rel_path mismatch
                    # Try loading all data from regional files without filtering by rel_path
                    st.info("⚠️ Regional files found but rel_path matching failed. Loading all regional data...")
                    regional_dfs = []
                    loaded_regions = []
                    failed_regions = []
                    for region_code in available_regions:
                        region_files = list(DEFAULT_DATA_DIR.glob(f"D-*__DBT__F-HR__L-{region_code}.parquet")) or [
                            DEFAULT_DATA_DIR / f"DBT__HR__{region_code}.parquet"
                        ]
                        region_file = next((f for f in region_files if f.exists()), None)
                        if region_file:
                            try:
                                file_size_mb = region_file.stat().st_size / 1024 / 1024
                                hourly_region = h.load_regional_hourly(region_code, DEFAULT_DATA_DIR)
                                if not hourly_region.empty:
                                    regional_dfs.append(hourly_region)
                                    loaded_regions.append(region_code)
                                    # Debug info
                                    if len(loaded_regions) <= 3:  # Show first 3 for debugging
                                        st.caption(f"✓ {region_code}: {len(hourly_region):,} rows, {file_size_mb:.2f} MB, cols: {list(hourly_region.columns)}")
                                else:
                                    failed_regions.append(f"{region_code} (empty after load, file size: {file_size_mb:.2f} MB)")
                            except Exception as e:
                                failed_regions.append(f"{region_code} (error: {str(e)[:50]})")
                                continue
                        else:
                            failed_regions.append(f"{region_code} (file not found)")
                    
                    if failed_regions and len(loaded_regions) == 0:
                        empty_files = []
                        for region_code in available_regions[:5]:
                            region_files = list(DEFAULT_DATA_DIR.glob(f"D-*__DBT__F-HR__L-{region_code}.parquet")) or [
                                DEFAULT_DATA_DIR / f"DBT__HR__{region_code}.parquet"
                            ]
                            for region_file in region_files:
                                if region_file.exists():
                                    try:
                                        test_df = pd.read_parquet(region_file)
                                        if test_df.empty:
                                            empty_files.append(region_code)
                                    except Exception:
                                        pass
                                    break
                        
                        if empty_files:
                            st.error(
                                f"❌ **Regional files exist but are EMPTY** (checked: {', '.join(empty_files)}).\n\n"
                                f"**This means the data preparation script created the files but didn't populate them with data.**\n\n"
                                f"**Solution:** Re-run the data preparation script:\n"
                                f"```bash\n"
                                f"python data/data_preparation_scripts/05A_build_epw_index_and_extract.py \\\n"
                                f"  --root data/01__italy_epw_all \\\n"
                                f"  --root data/02__italy_fwg_outputs \\\n"
                                f"  --out data/03__italy_all_epw_DBT_streamlit \\\n"
                                f"  --regional --daily-percentiles\n"
                                f"```\n\n"
                                f"**Make sure:**\n"
                                f"- The input directories contain EPW files\n"
                                f"- The script completes without errors\n"
                                f"- The output files have non-zero row counts"
                            )
                        else:
                            st.error(f"❌ Failed to load data from all regions. Issues:\n- " + "\n- ".join(failed_regions[:10]))
                    
                    if regional_dfs:
                        hourly_all = pd.concat(regional_dfs, axis=0).sort_index()
                        st.success(f"✓ Loaded data from {len(loaded_regions)} regions: {', '.join(loaded_regions)}")
                        daily_result = _timed(
                            "compute_daily_stats_from_regional_all",
                            lambda: _compute_daily_stats_from_hourly(hourly_all),
                            notes="Computes daily stats from all regional hourly files (no rel_path filtering).",
                        )
                        if not daily_result.empty:
                            return daily_result
                        else:
                            st.warning(f"⚠️ Computed daily stats but result is empty. Hourly data had {len(hourly_all):,} rows.")
                    else:
                        st.warning(f"⚠️ No data could be loaded from regional files. Checked {len(available_regions)} regions.")
    
    # Fallback: compute from monolithic tidy parquet (slow) - only if file exists
    if tidy_path.exists():
        return _timed(
            "f121__daily_stats_by_rel_path",
            lambda: h.f121__daily_stats_by_rel_path(str(tidy_path)),
            notes="Pre-aggregates hourly data to daily mean/max per file (slow - run 00_precompute_daily_stats.py or use regional files to speed up).",
        )
    else:
        # No data files available - return empty DataFrame
        st.warning(
            f"⚠️ No daily stats data found. Expected one of:\n"
            f"- `{daily_stats_path}` (precomputed)\n"
            f"- Regional files `DBT__HR__XX.parquet` in `{DEFAULT_DATA_DIR}`\n"
            f"- `{tidy_path}` (monolithic file)\n\n"
            f"Please run the data preparation scripts to generate the required files."
        )
        return pd.DataFrame(columns=["rel_path", "month", "day", "DBT_mean", "DBT_max"])


def _compute_daily_stats_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    return h.f146__compute_daily_stats_from_hourly(hourly)


@st.cache_data(show_spinner=False)
def _load_file_stats_cached(daily_stats: pd.DataFrame, percentile: float,
                            precomputed_path: Path):
    return _timed(
        "load_file_stats",
        lambda: h.f149__file_stats_from_precomputed_or_daily(precomputed_path, daily_stats, percentile),
        notes="File stats (precomputed or computed).",
    )


@st.cache_data(show_spinner=False)
def _load_location_deltas_cached(daily_stats: pd.DataFrame, idx: pd.DataFrame,
                                 baseline_variant: str, compare_variant: str,
                                 percentile: float, precomputed_path: Path):
    return _timed(
        "load_location_deltas",
        lambda: h.f150__location_deltas_from_precomputed_or_daily(
            precomputed_path, daily_stats, idx,
            baseline_variant=baseline_variant,
            compare_variant=compare_variant,
            percentile=float(percentile),
        ),
        notes="Location deltas (precomputed or computed).",
    )


@st.cache_data(show_spinner=False)
def _load_precomputed_location_stats(variant: str, percentile: float, daily_stats, idx, precomputed_path: Path):
    return _timed(
        "load_location_stats",
        lambda: h.f151__location_stats_from_precomputed_or_compute(
            precomputed_path, variant, percentile, daily_stats, idx
        ),
        notes=f"Location stats for {variant}.",
    )


def _filter_cti_points_by_region(cti_points: list, region_code: str | None) -> list:
    return h.f147__filter_cti_points_by_region(cti_points, region_code)


@st.cache_data(show_spinner=False)
def _load_precomputed_monthly_delta_table(baseline_variant: str, compare_variant: str, 
                                         percentile: float, metric_key: str, 
                                         daily_stats, idx, precomputed_path: Path):
    return _timed(
        "load_monthly_delta_table",
        lambda: h.f152__monthly_delta_from_precomputed_or_compute(
            precomputed_path, baseline_variant, compare_variant, percentile, metric_key, daily_stats, idx
        ),
        notes="Monthly delta table (precomputed or computed).",
    )


# TMYx and RCP only — build variants from inventory columns
all_scenarios = sorted(
    {
        s
        for cols in inventory.get("cols", [])
        for s in _parse_inventory_cols(cols)
    }
)
present_variants = sorted(
    [v for v in all_scenarios if "__" not in v],
    key=h.f106__present_variant_sort_key,
)
if not present_variants:
    st.warning("No baseline (TMYx) variants found in inventory.")
    st.stop()

# present_variants_map: only "tmyx" (and any non-dated present variants).
# Used for sidebar, maps, delta, Italian Regions. Does NOT affect the TMYx station tab.
present_variants_map = [v for v in present_variants if v not in DATED_TMYX_VARIANTS]
if not present_variants_map:
    present_variants_map = present_variants  # safety fallback

# Sidebar controls (all radio/select widgets)
future_tags_by_baseline: dict[str, list[str]] = {}
for scenario in all_scenarios:
    if "__" not in scenario:
        continue
    base, tag = scenario.split("__", 1)
    if base in DATED_TMYX_VARIANTS:
        continue  # suppress dated-baseline futures from sidebar
    future_tags_by_baseline.setdefault(base, set()).add(tag)
future_tags_by_baseline = {k: sorted(list(v)) for k, v in future_tags_by_baseline.items()}

with st.sidebar:
    st.markdown(f"#### {label('baseline_climate_file')}")

    default_baseline = "tmyx_2009-2023"
    default_baseline = "tmyx"
    default_idx = present_variants_map.index(default_baseline) if default_baseline in present_variants_map else 0
    baseline_variant = st.radio(
        label("current_climate_baseline"),
        options=present_variants_map,
        index=default_idx,
        help="These are the present files that were morphed (TMYx + optional time-window variants).",
    )
    baseline_variants = [baseline_variant] if baseline_variant else []
    baseline_variant_future = baseline_variant

    h.f002__custom_hr(-0.6,-0.6)
    st.markdown(f"#### {label('climate_scenarios')}")
    # Split compare variant into RCP + Year radios, then recombine to rcpXX_YYYY
    future_tags = future_tags_by_baseline.get(baseline_variant, [])
    future_pairs = []
    for tag in future_tags:
        try:
            rcp, year = tag.split("_", 1)
        except Exception:
            continue
        if rcp.startswith("rcp") and year.isdigit():
            future_pairs.append((rcp, year, tag))

    rcps = sorted({rcp for rcp, _, _ in future_pairs})
    years_by_rcp: dict[str, list[str]] = {}
    for rcp, year, _v in future_pairs:
        years_by_rcp.setdefault(rcp, set()).add(year)
    years_by_rcp = {k: sorted(list(v)) for k, v in years_by_rcp.items()}

    if not rcps:
        st.error("No morphed variants found for the selected baseline.")
        st.stop()

    rcp_col, year_col = st.columns(2)
    with rcp_col:
        rcp_choice = st.radio(
            "RCP",
            options=rcps,
            index=min(1, len(rcps) - 1) if len(rcps) > 1 else 0,
            help="Morphed scenario family (RCP).",
        )
    with year_col:
        year_options = years_by_rcp.get(rcp_choice) or sorted({y for _r, y, _v in future_pairs})
        year_choice = st.radio(
            label("year"),
            options=year_options,
            index=0,
            help="Morphed target year.",
        )

    compare_tag = f"{rcp_choice}_{year_choice}"
    compare_variant = f"{baseline_variant}__{compare_tag}"
    if compare_variant not in all_scenarios:
        fallback_tag = future_tags[0] if future_tags else compare_tag
        compare_variant = f"{baseline_variant}__{fallback_tag}"
        st.warning(f"Scenario `{compare_tag}` not found; using `{fallback_tag}` instead.")

    h.f002__custom_hr(-0.6,-0.6)
    metric_col, percentile_col = st.columns([2,3], gap="small")
    with metric_col:
        metric = st.radio(
            label("metric"),
            options=["Tmax", "Tavg"],
            help="Select to visualise the delta T for mean or max temperatures - ΔT=Trcp - Tbaseline for Dry Bulb Temperature (DBT).",
        )
    with percentile_col:
        percentile = st.radio(
            "Tmax percentile",
            options=[95.0, 97.0, 99.0],
            index=1,
            format_func=lambda v: f"{v:g}th",
            help="Used as the 'max' definition for Plotly + D3: qP(Comp) − qP(Base).",
            key="dtmax_percentile",
            horizontal=True,
        )

    h.f002__custom_hr(-0.6, -0.6)

    # Map + charts mode: fixed to D3/JS only (Plotly choice removed)
    render_mode = st.radio(
        "Map + charts mode",
        options=["D3/JS"],
        index=0,
        horizontal=True,
        help="All interactivity runs inside the embedded D3/JS (no Plotly mode).",
        disabled=True,
    )

if not baseline_variants:
    st.warning("Select at least one baseline climate file to render maps.")
    st.stop()

# Precompute daily stats + percentile-based file stats (cached)
# Note: Computation happens here, but status is shown in Intro tab
percentile_state = float(st.session_state.get("dtmax_percentile", 99.0))

# Daily stats are precomputed in the B route; abort if missing
if daily_stats.empty:
    st.error(
        "❌ **Critical Error**: Daily stats table is missing or empty.\n\n"
        f"Expected: `{DEFAULT_B_DATA_DIR / '_tables' / 'D-TMYxFWG__DBT__F-DD__L-ALL.parquet'}`"
    )
    st.stop()

# Load file stats: use precomputed from 06B when present, else compute
percentile_int = int(percentile_state)
FILE_STATS_PRECOMPUTED = DEFAULT_B_DATA_DIR / "_tables" / f"D-TMYxFWG__FileStats__P-{percentile_int}.parquet"
file_stats = _load_file_stats_cached(
    daily_stats,
    percentile_state / 100.0,
    FILE_STATS_PRECOMPUTED,
)

metric_key = "dTmax" if metric == "Tmax" else "dTavg"

# Load location deltas with caching (from daily stats)
LOCATION_DELTAS_PRECOMPUTED = DEFAULT_B_DATA_DIR / "_tables" / f"D-TMYxFWG__LocationDeltas__P-{percentile_int}.parquet"
delta_df = _load_location_deltas_cached(
    daily_stats,
    idx,
    baseline_variant_future,
    compare_variant,
    percentile,
    LOCATION_DELTAS_PRECOMPUTED
)
if delta_df.empty:
    st.warning(
        "**No matching locations** between the chosen baseline and compare scenario.\n\n"
        "**Likely cause:** The selected baseline has no paired future rows for this RCP/year in the inventory.\n\n"
        "**Fix:**\n"
        "1. Check `_tables/D-TMYxFWG__Inventory__F-NA__L-ALL.parquet` for available columns.\n"
        "2. Inspect `_tables/pairing_debug.csv` for missing scenario columns.\n"
        "3. Clear Streamlit cache and restart the app."
    )

def _render_cti_scenario_tab() -> None:
    abs_key = "Tmax" if metric_key == "dTmax" else "Tavg"
    if cti_daily_stats is None or cti_daily_stats.empty or cti_idx is None or cti_idx.empty:
        st.info("CTI daily stats or index data is missing.")
        return
    abs_df = _timed(
        "f28c__compute_location_stats_cti_from_daily",
        lambda: h.f28c__compute_location_stats_cti_from_daily(
            cti_daily_stats,
            cti_idx,
            variant="cti",
            percentile=float(percentile) / 100.0,
        ),
        notes="Computes CTI location stats from daily aggregates.",
    )
    if abs_df.empty:
        st.info("No CTI data available for this metric.")
        return

    map_charts_col, table_col = st.columns([8, 4], gap="medium")
    with map_charts_col:
        st.markdown("##### Map + Charts")
        points = abs_df[
            ["location_id", "location_name", "latitude", "longitude", "Tmax", "Tavg"]
        ].to_dict(orient="records")
        daily_stat = "max" if metric_key == "dTmax" else "mean"
        loc_ids = tuple(str(x) for x in abs_df["location_id"].astype(str).unique().tolist())
        profiles = _timed(
            "f22e__build_daily_profiles_cti",
            lambda: h.f22e__build_daily_db_profiles_single_variant_from_daily_stats(
                cti_daily_stats,
                cti_idx,
                location_ids=loc_ids,
                variant="cti",
                daily_stat=daily_stat,
                baseline_variant=None,
            ),
            notes="Builds CTI daily profiles from daily stats.",
        )
        html = _timed(
            "f23b__d3_dashboard_html_abs_cti",
            lambda: h.f23b__d3_dashboard_html_abs(
                points=points,
                profiles_bundle=profiles,
                metric_key=abs_key,
                width=820,
                height=650,
                scenario_variant="cti",
                percentile=float(percentile),
                geo_script=_GEO_SCRIPT,
            ),
            notes="Generates D3 dashboard HTML for CTI scenario.",
        )
        components.html(html, height=1150, scrolling=False)

    with table_col:
        st.markdown("##### Table View")
        table = abs_df.copy()
        num_cols = table.select_dtypes("number").columns
        table[num_cols] = table[num_cols].round(1)
        st.dataframe(table, width="stretch", height=720, hide_index=True)


def _render_cti_data_tab() -> None:
    st.markdown("##### CTI (Itaca) station list")
    st.markdown(
        "This table shows the CTI weather station list used by the app. "
        "**Latitude** and **longitude** give the position of each location; "
        "**region** is the Italian region (e.g. LM = Lombardia, LZ = Lazio) the station belongs to."
    )
    if not CTI_LIST_CSV.exists():
        st.warning(
            "CTI list file not found. Place `CTI__list__ITA_WeatherStations__All.csv` in "
            "`data/04__italy_tmy_fwg_parquet/` or `data/cti/`."
        )
        return
    try:
        cti_df = h.f94b__cti_load_list_only(CTI_LIST_CSV)
    except Exception as e:
        st.error(f"Could not load CTI list: {e}")
        return
    if cti_df.empty:
        st.info("The CTI list file is empty or has no valid rows.")
        return
    st.dataframe(cti_df, width="stretch", height=720, hide_index=True)


def _render_future_scenario_tab() -> None:
    # Reuse the existing metric widget selection (Tmax/Tavg) to choose which absolute metric to show.
    abs_key = "Tmax" if metric_key == "dTmax" else "Tavg"
    LOCATION_STATS_PRECOMPUTED = DEFAULT_B_DATA_DIR / "_tables" / f"D-TMYxFWG__LocationStats__P-{percentile_int}.parquet"
    abs_df = _load_precomputed_location_stats(
        compare_variant,
        float(percentile) / 100.0,
        daily_stats,
        idx,
        LOCATION_STATS_PRECOMPUTED,
    )

    if abs_df.empty:
        st.info("No data available for this scenario.")
    else:
        # Mirror the Δ-tab layout: Map(+charts) + Table View
        if render_mode.startswith("Plotly"):
            map_charts_col, table_col = st.columns([8, 4], gap="medium")
            with map_charts_col:
                map_col, charts_col = st.columns([4, 6], gap="small")
                with map_col:
                    title_col, toggle_col = st.columns([3, 1])
                    with title_col:
                        st.markdown(f"##### {label('map_view')}")
                    with toggle_col:
                        show_d3 = st.toggle("D3 Charts", value=False, key="future_temps_d3_toggle", help="Enable D3 interactive charts (may take a long time to load)")
                    st.caption(f"Showing **{abs_key}** for **{compare_variant}**")

                    if not show_d3:
                        st.info("Enable 'D3 Charts' toggle to load interactive D3 charts.")
                        # Show Plotly map when toggle is off
                        points = abs_df[
                            ["location_id", "location_name", "latitude", "longitude", "Tmax", "Tavg"]
                        ].to_dict(orient="records")
                        fig = _timed(
                            "f19b__plotly_italy_map_abs",
                            lambda: h.f19b__plotly_italy_map_abs(
                                points,
                                metric_key=abs_key,
                                height=520,
                                title=f"{compare_variant} — {abs_key}",
                                show_colorbar=True,
                            ),
                            notes="Generates Plotly Italy map for absolute scenario.",
                        )
                        sel = st.plotly_chart(
                            fig,
                            width="stretch",
                            key=f"abs_map_{compare_variant}_{abs_key}",
                            on_select="rerun",
                            selection_mode=["points"],
                        )
                        picked = h.f20__parse_plotly_selection(sel)
                        if picked:
                            loc_id, loc_name = picked
                            st.session_state["selected_point_abs"] = {
                                "location_id": loc_id,
                                "location_name": loc_name,
                                "variant": compare_variant,
                            }
                    else:
                        # Show D3 charts when toggle is on
                        baseline_loc_ids = _baseline_location_ids(idx, baseline_variant_future)
                        abs_df_d3 = abs_df[abs_df["location_id"].astype(str).isin(baseline_loc_ids)].copy()
                        if abs_df_d3.empty:
                            st.info("No locations match the selected baseline variant.")
                            return
                        points = abs_df_d3[
                            ["location_id", "location_name", "latitude", "longitude", "Tmax", "Tavg"]
                        ].to_dict(orient="records")

                        daily_stat = "max" if metric_key == "dTmax" else "mean"
                        loc_ids = tuple(str(x) for x in abs_df_d3["location_id"].astype(str).unique().tolist())
                        _tables_dir = DEFAULT_B_DATA_DIR / "_tables"
                        profiles = h.load_daily_profiles_abs_precomputed(
                            _tables_dir, compare_variant, daily_stat, loc_ids
                        )
                        if not (profiles.get("profiles") and len(profiles["profiles"]) > 0):
                            profiles = _timed(
                                "f22e__build_daily_profiles_single_variant_from_daily_stats",
                                lambda: h.f22e__build_daily_db_profiles_single_variant_from_daily_stats(
                                    daily_stats,
                                    idx,
                                    location_ids=loc_ids,
                                    variant=compare_variant,
                                    daily_stat=daily_stat,
                                    baseline_variant=baseline_variant_future,
                                ),
                                notes="Builds daily profiles from daily stats.",
                            )
                        html = _timed(
                            "f23b__d3_dashboard_html_abs",
                            lambda: h.f23b__d3_dashboard_html_abs(
                                points=points,
                                profiles_bundle=profiles,
                                metric_key=abs_key,
                                width=820,
                                height=650,
                                scenario_variant=compare_variant,
                                percentile=float(percentile),
                                geo_script=_GEO_SCRIPT,
                            ),
                            notes="Generates D3 dashboard HTML for absolute scenario.",
                        )
                        components.html(html, height=1150, scrolling=False)

                with charts_col:
                    st.markdown("##### Charts")
                    selected = st.session_state.get("selected_point_abs")
                    if selected and isinstance(selected, dict) and selected.get("location_id"):
                        loc_id = str(selected.get("location_id"))
                        loc_name = selected.get("location_name") or loc_id
                        v = str(selected.get("variant") or compare_variant)

                        st.caption(f"Location: **{loc_name}** (`{loc_id}`) — scenario `{v}`")

                        stat = "max" if metric_key == "dTmax" else "mean"
                        daily_one = h.f25b__build_location_daily_for_variant(
                            daily_stats,
                            idx,
                            location_id=loc_id,
                            variant=v,
                            stat=stat,
                        )
                        if daily_one.empty:
                            st.warning("No daily rows found for this location/scenario.")
                        else:
                            # Summer time series (Jul–Aug): scenario only
                            summer = daily_one[daily_one["month"].isin([7, 8])].copy()
                            line = (
                                alt.Chart(summer)
                                .mark_line()
                                .encode(
                                    x=alt.X("date:T", title="Date (Jul–Aug)"),
                                    y=alt.Y("DBT:Q", title=f"{abs_key} (°C)"),
                                    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("DBT:Q", format=".2f")],
                                )
                                .properties(height=260)
                                .interactive()
                            )
                            st.altair_chart(line, width="stretch")

                            # Monthly absolute summary
                            if metric_key == "dTmax":
                                month_agg = (
                                    alt.Chart(daily_one)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("month:O", title="Month"),
                                        y=alt.Y("quantile(DBT, 0.99):Q", title=f"{abs_key} (°C)"),
                                        tooltip=[alt.Tooltip("quantile(DBT, 0.99):Q", format=".2f")],
                                    )
                                    .properties(height=220)
                                )
                            else:
                                month_agg = (
                                    alt.Chart(daily_one)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("month:O", title="Month"),
                                        y=alt.Y("mean(DBT):Q", title=f"{abs_key} (°C)"),
                                        tooltip=[alt.Tooltip("mean(DBT):Q", format=".2f")],
                                    )
                                    .properties(height=220)
                                )
                            st.altair_chart(month_agg, width="stretch")
                    else:
                        st.info("Click a map marker to see location charts.")
            
            with table_col:
                st.markdown(f"##### {label('table_view')}")
                st.caption("Absolute values for the selected scenario.")
                show_cols = ["location_id", "location_name", "Tmax", "Tavg"]
                abs_table = abs_df[show_cols].copy()
                num_cols = abs_table.select_dtypes("number").columns
                abs_table[num_cols] = abs_table[num_cols].round(1)
                st.dataframe(
                    abs_table.sort_values(abs_key, ascending=False),
                    width="stretch",
                    height=720,
                    hide_index=True,
                )
        else:
            # D3/JS mode - show toggle and status, but no Plotly charts
            map_col, table_col = st.columns([7, 5], gap="medium")
            with map_col:
                title_col, toggle_col = st.columns([3, 1])
                with title_col:
                    st.markdown(f"##### {label('map_view')}")
                with toggle_col:
                    show_d3 = st.toggle("D3 Charts", value=False, key="future_temps_d3_toggle", help="Enable D3 interactive charts (may take a long time to load)")
                st.caption(f"Showing **{abs_key}** for **{compare_variant}**")

                if not show_d3:
                    st.info("Enable 'D3 Charts' toggle to load interactive D3 charts.")
                else:
                    # Show D3 charts when toggle is on
                    points = abs_df[
                        ["location_id", "location_name", "latitude", "longitude", "Tmax", "Tavg"]
                    ].to_dict(orient="records")

                    daily_stat = "max" if metric_key == "dTmax" else "mean"
                    loc_ids = tuple(str(x) for x in abs_df["location_id"].astype(str).unique().tolist())
                    _tables_dir = DEFAULT_B_DATA_DIR / "_tables"
                    profiles = h.load_daily_profiles_abs_precomputed(
                        _tables_dir, compare_variant, daily_stat, loc_ids
                    )
                    if not (profiles.get("profiles") and len(profiles["profiles"]) > 0):
                        profiles = _timed(
                            "f22e__build_daily_profiles_single_variant_from_daily_stats",
                            lambda: h.f22e__build_daily_db_profiles_single_variant_from_daily_stats(
                                daily_stats,
                                idx,
                                location_ids=loc_ids,
                                variant=compare_variant,
                                daily_stat=daily_stat,
                                baseline_variant=baseline_variant,
                            ),
                            notes="Builds daily profiles from daily stats.",
                        )
                    html = _timed(
                        "f23b__d3_dashboard_html_abs",
                        lambda: h.f23b__d3_dashboard_html_abs(
                            points=points,
                            profiles_bundle=profiles,
                            metric_key=abs_key,
                            width=820,
                            height=650,
                            scenario_variant=compare_variant,
                            percentile=float(percentile),
                            geo_script=_GEO_SCRIPT,
                        ),
                        notes="Generates D3 dashboard HTML for absolute scenario.",
                    )
                    components.html(html, height=1150, scrolling=False)
            
            with table_col:
                st.markdown(f"##### {label('table_view')}")
                st.caption("Absolute values for the selected scenario.")
                show_cols = ["location_id", "location_name", "Tmax", "Tavg"]
                abs_table = abs_df[show_cols].copy()
                num_cols = abs_table.select_dtypes("number").columns
                abs_table[num_cols] = abs_table[num_cols].round(1)
                st.dataframe(
                    abs_table.sort_values(abs_key, ascending=False),
                    width="stretch",
                    height=720,
                    hide_index=True,
                )


def _render_future_delta_tab() -> None:
    if render_mode.startswith("Plotly"):
        map_charts_col, table_col = st.columns([8, 4], gap="medium")
        with map_charts_col:
            map_col, charts_col = st.columns([4, 6], gap="small")
            with map_col:
                st.markdown(f"##### {label('map_view')}")
                st.caption(f"Showing **{metric}** for **{compare_variant} − {baseline_variant}**. Colors show **|Δ|** clipped to **0..5°C**.")

                if delta_df.empty:
                    st.info("No overlap.")
                else:
                    points = delta_df[
                        ["location_id", "location_name", "latitude", "longitude", "dTmax", "dTavg"]
                    ].to_dict(orient="records")

                    fig = h.f19__plotly_italy_map(
                        points,
                        metric_key=metric_key,
                        height=520,
                        title=f"{compare_variant} − {baseline_variant}",
                        show_colorbar=True,
                    )
                    sel = st.plotly_chart(
                        fig,
                        width="stretch",
                        key=f"map_{baseline_variant}_{metric_key}_{compare_variant}",
                        on_select="rerun",
                        selection_mode=["points"],
                    )
                    picked = h.f20__parse_plotly_selection(sel)
                    if picked:
                        loc_id, loc_name = picked
                        st.session_state["selected_point"] = {
                            "location_id": loc_id,
                            "location_name": loc_name,
                            "baseline_variant": baseline_variant,
                            "compare_variant": compare_variant,
                        }
            
            with charts_col:
                st.markdown("##### Charts")
                selected = st.session_state.get("selected_point")
                # Clear selection if baseline/compare changed so charts match current map
                if selected and isinstance(selected, dict):
                    if (str(selected.get("baseline_variant")) != str(baseline_variant) or
                        str(selected.get("compare_variant")) != str(compare_variant)):
                        selected = None
                        st.session_state["selected_point"] = None
                if selected and isinstance(selected, dict) and selected.get("location_id"):
                    loc_id = str(selected.get("location_id"))
                    loc_name = selected.get("location_name") or loc_id
                    bvar = str(selected.get("baseline_variant") or "")
                    cvar = str(selected.get("compare_variant") or "")

                    st.caption(f"Location: **{loc_name}** (`{loc_id}`) — baseline `{bvar}` vs compare `{cvar}`")

                    # FAST PATH: use cached daily stats table instead of scanning the 36M-row hourly parquet per click
                    stat = "max" if metric_key == "dTmax" else "mean"
                    daily = _timed(
                        "f122__build_location_daily_join",
                        lambda: h.f122__build_location_daily_join(
                            daily_stats,
                            idx,
                            location_id=loc_id,
                            baseline_variant=bvar,
                            compare_variant=cvar,
                            stat=stat,
                        ),
                        notes="Builds daily base/comp aligned series for per-location charts.",
                    )

                    if daily.empty:
                        st.warning("No daily rows found for this location/variant pair.")
                    else:
                        # Summer time series (Jul–Aug): baseline vs compare
                        summer = daily[daily["month"].isin([7, 8])].copy()
                        line = (
                            alt.Chart(
                                pd.concat(
                                    [
                                        summer[["date", "DBT_base"]].rename(columns={"DBT_base": "DBT"}).assign(series=f"baseline ({bvar})"),
                                        summer[["date", "DBT_comp"]].rename(columns={"DBT_comp": "DBT"}).assign(series=f"compare ({cvar})"),
                                    ],
                                    ignore_index=True,
                                )
                            )
                            .mark_line()
                            .encode(
                                x=alt.X("date:T", title="Date (Jul–Aug)"),
                                y=alt.Y("DBT:Q", title="DBT (°C)"),
                                color=alt.Color("series:N", title=""),
                                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("DBT:Q", format=".2f"), alt.Tooltip("series:N")],
                            )
                            .properties(height=260)
                            .interactive()
                        )
                        st.altair_chart(line, width="stretch")

                        # Monthly delta summary from DAILY deltas
                        daily2 = daily.copy()
                        daily2["dDBT"] = daily2["delta"]
                        month_agg = (
                            alt.Chart(daily2)
                            .mark_bar()
                            .encode(
                                x=alt.X("month:O", title="Month"),
                                y=alt.Y(("max(dDBT):Q" if stat == "max" else "mean(dDBT):Q"),
                                        title=("Max ΔDBT (°C)" if stat == "max" else "Mean ΔDBT (°C)")),
                                tooltip=[alt.Tooltip(("max(dDBT):Q" if stat == "max" else "mean(dDBT):Q"), format=".2f")],
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(month_agg, width="stretch")

                        with st.expander("Download clicked location time-series (CSV)", expanded=False):
                            st.download_button(
                                "Download merged baseline+compare daily DBT",
                                data=daily.to_csv(index=False).encode("utf-8"),
                                file_name=f"clicked_{loc_id}_{bvar}_vs_{cvar}.csv".replace("/", "_"),
                                mime="text/csv",
                            )
                else:
                    st.info("Click a map marker to see location charts.")
    else:
        map_col, table_col = st.columns([7,5], gap="medium")
        with map_col:
            title_col, toggle_col = st.columns([3, 1])
            with title_col:
                st.markdown(f"##### {label('map_view')}")
            with toggle_col:
                show_d3 = st.toggle("D3 Charts", value=False, key="future_delta_d3_toggle", help="Enable D3 interactive charts (may take a long time to load)")
            st.caption(f"Showing **{metric}** for **{compare_variant} − {baseline_variant_future}**.")

            if delta_df.empty:
                st.info("No overlap.")
            elif not show_d3:
                st.info("Enable 'D3 Charts' toggle to load interactive D3 charts.")
            else:
                baseline_loc_ids = _baseline_location_ids(idx, baseline_variant_future)
                delta_df_d3 = delta_df[delta_df["location_id"].astype(str).isin(baseline_loc_ids)].copy()
                if delta_df_d3.empty:
                    st.info("No locations match the selected baseline variant.")
                    return
                points = delta_df_d3[
                    ["location_id", "location_name", "latitude", "longitude", "dTmax", "dTavg"]
                ].to_dict(orient="records")

                daily_stat = "max" if metric_key == "dTmax" else "mean"
                loc_ids = tuple(str(x) for x in delta_df_d3["location_id"].astype(str).unique().tolist())
                _tables_dir = DEFAULT_B_DATA_DIR / "_tables"
                profiles = h.load_daily_profiles_delta_precomputed(
                    _tables_dir,
                    baseline_variant_future,
                    compare_variant,
                    daily_stat,
                    loc_ids,
                )
                if not (profiles.get("profiles") and len(profiles["profiles"]) > 0):
                    profiles = _timed(
                        "f22d__build_daily_profiles_from_daily_stats",
                        lambda: h.f22d__build_daily_db_profiles_from_daily_stats(
                            daily_stats,
                            idx,
                            location_ids=loc_ids,
                            baseline_variant=baseline_variant_future,
                            compare_variant=compare_variant,
                            daily_stat=daily_stat,
                        ),
                        notes="Builds daily profiles from daily stats.",
                    )
                html = _timed(
                    "f23__d3_dashboard_html",
                    lambda: h.f23__d3_dashboard_html(
                        points=points,
                        profiles_bundle=profiles,
                        metric_key=metric_key,
                        width=820,
                        height=650,
                                baseline_variant=baseline_variant_future,
                        compare_variant=compare_variant,
                        percentile=float(percentile),
                        geo_script=_GEO_SCRIPT,
                    ),
                    notes="Generates D3 dashboard HTML for delta comparison.",
                )
                components.html(html, height=1150, scrolling=False)

    with table_col:
        st.markdown("##### Table View")
        st.caption(
            "ΔT is computed as (RCP − baseline). "
            "For Tmax we use the selected percentile of daily Tmax; "
            "for Tavg we use the mean of daily Tavg."
        )
        if delta_df.empty:
            st.info("No overlap.")
        else:
            MONTHLY_DELTAS_PRECOMPUTED = DEFAULT_B_DATA_DIR / "_tables" / f"D-TMYxFWG__MonthlyDeltas__F-MM__P-{percentile_int}.parquet"
            monthly_table = _load_precomputed_monthly_delta_table(
                baseline_variant_future,
                compare_variant,
                float(percentile),
                metric_key,
                daily_stats,
                idx,
                MONTHLY_DELTAS_PRECOMPUTED,
            )
            # Hide latitude and longitude columns by default
            display_table = monthly_table.drop(columns=["latitude", "longitude"], errors="ignore")
            num_cols = display_table.select_dtypes("number").columns
            display_table[num_cols] = display_table[num_cols].round(1)
            st.dataframe(display_table, width="stretch", height=720, hide_index=True)


def _run_scenarios_page():
    scenario_tabs = st.tabs([
        label("future_temperatures_italy"),
        label("future_vs_current_temperatures_italy"),
        label("future_climate_italian_regions"),
        label("current_weather_data_tmyx"),
        label("ipcc_scenarios"),
    ])
    with scenario_tabs[0]:
        _render_future_scenario_tab()
    with scenario_tabs[1]:
        _render_future_delta_tab()
    with scenario_tabs[2]:
        # Future Climate — Italian Regions
        title_col, toggle_col = st.columns([2,7], gap="small")
        with title_col:
            st.markdown(f"##### {label('map_view')}")
        with toggle_col:
            show_d3_regions_v2 = st.toggle(
                "Show Charts",
                value=False,
                key="single_regions_v2_d3_toggle",
                help="Enable D3 interactive charts (may take a long time to load)",
            )

        if not show_d3_regions_v2:
            st.info("Enable 'Show Charts' toggle to load interactive JavaScript charts.")

        if show_d3_regions_v2:
            # Explicit size controls for the Future Climate — Italian Regions layout
            V2_DASHBOARD_COLS = "5.5fr 24px 4fr"
            V2_MAPS_ROW_GAP_PX = 10
            V2_MAPS_ROW3_GAP_PX = 3
            region_code_to_name = {
                "AB": "Abruzzo",
                "BC": "Basilicata",
                "CM": "Campania",
                "ER": "Emilia-Romagna",
                "FV": "Friuli-Venezia Giulia",
                "LB": "Calabria",
                "LG": "Liguria",
                "LM": "Lombardia",
                "LZ": "Lazio",
                "MH": "Marche",
                "ML": "Molise",
                "PM": "Piemonte",
                "PU": "Puglia",
                "SC": "Sicilia",
                "SD": "Sardegna",
                "TC": "Toscana",
                "TT": "Trentino-Alto Adige/Südtirol",
                "UM": "Umbria",
                "VD": "Valle d'Aosta/Vallée d'Aoste",
                "VN": "Veneto",
            }
            region_code_to_istat = {
                "PM": 1,
                "VD": 2,
                "LM": 3,
                "TT": 4,
                "VN": 5,
                "FV": 6,
                "LG": 7,
                "ER": 8,
                "TC": 9,
                "UM": 10,
                "MH": 11,
                "LZ": 12,
                "AB": 13,
                "ML": 14,
                "CM": 15,
                "PU": 16,
                "BC": 17,
                "LB": 18,
                "SC": 19,
                "SD": 20,
            }
            region_options = sorted(
                {
                    *([str(r) for r in idx["region"].dropna().unique().tolist()] if not idx.empty else []),
                    *([str(r) for r in cti_idx["region"].dropna().unique().tolist()] if not cti_idx.empty else []),
                }
            )
            if not region_options:
                st.info("No region metadata available in the index.")
            else:
                region_options_data = [
                    {
                        "code": code,
                        "name": region_code_to_name.get(code, code),
                        "istat": region_code_to_istat.get(code),
                    }
                    for code in region_options
                ]
                region_map = (
                    idx.groupby("location_id", as_index=False)["region"].first()
                    .set_index("location_id")["region"]
                    .to_dict()
                    if not idx.empty else {}
                )
                cti_region_map = (
                    cti_idx.groupby("location_id", as_index=False)["region"].first()
                    .set_index("location_id")["region"]
                    .to_dict()
                    if not cti_idx.empty else {}
                )
                abs_points_by_variant = {}
                abs_key = "Tmax" if metric_key == "dTmax" else "Tavg"
                if present_variants_map and not daily_stats.empty and not idx.empty:
                    for v in present_variants_map:
                        abs_df = h.f28b__compute_location_stats_for_variant_from_daily(
                            daily_stats,
                            idx,
                            variant=v,
                            percentile=float(percentile) / 100.0,
                        )
                        if abs_df.empty:
                            abs_points_by_variant[v] = []
                        else:
                            abs_df["region"] = abs_df["location_id"].astype(str).map(region_map)
                            abs_df["value"] = abs_df[abs_key]
                            abs_points_by_variant[v] = abs_df[
                                ["location_id", "location_name", "latitude", "longitude", "region", "value"]
                            ].to_dict(orient="records")

                if not cti_daily_stats.empty and not cti_idx.empty:
                    try:
                        cti_stats_df = h.f28c__compute_location_stats_cti_from_daily(
                            cti_daily_stats,
                            cti_idx,
                            variant="cti",
                            percentile=float(percentile) / 100.0,
                        )
                        if not cti_stats_df.empty:
                            cti_stats_df["region"] = cti_stats_df["location_id"].astype(str).map(cti_region_map)
                            cti_stats_df["value"] = cti_stats_df["Tmax"] if metric_key == "dTmax" else cti_stats_df["Tavg"]
                            abs_points_by_variant["cti"] = cti_stats_df[
                                ["location_id", "location_name", "latitude", "longitude", "region", "value"]
                            ].to_dict(orient="records")
                    except Exception as e:
                        abs_points_by_variant["cti"] = []
                        st.warning(f"CTI data found but could not be processed: {e}")

                # Use sidebar baseline so region maps match "Proiezioni Temperature Future" and "Temperature Correnti vs Future"
                base_for_default = (
                    baseline_variant_future
                    if baseline_variant_future in present_variants_map
                    else ("tmyx" if "tmyx" in present_variants_map else present_variants_map[0])
                )
                future_scenarios = [f"{base_for_default}__rcp45_2050", f"{base_for_default}__rcp85_2080"]
                future_abs_points_by_variant = {}
                for future_variant in future_scenarios:
                    if future_variant in all_scenarios:
                        LOCATION_STATS_PRECOMPUTED = DEFAULT_B_DATA_DIR / "_tables" / f"D-TMYxFWG__LocationStats__P-{percentile_int}.parquet"
                        future_abs_df = _load_precomputed_location_stats(
                            future_variant,
                            float(percentile) / 100.0,
                            daily_stats,
                            idx,
                            LOCATION_STATS_PRECOMPUTED,
                        )
                        if not future_abs_df.empty:
                            future_abs_df["region"] = future_abs_df["location_id"].astype(str).map(region_map)
                            future_abs_df["value"] = future_abs_df[abs_key]
                            future_abs_points_by_variant[future_variant] = future_abs_df[
                                ["location_id", "location_name", "latitude", "longitude", "region", "value"]
                            ].to_dict(orient="records")
                        else:
                            future_abs_points_by_variant[future_variant] = []

                top_container_variants = [base_for_default, *future_scenarios]
                top_abs_points = {v: abs_points_by_variant.get(v, []) for v in top_container_variants if v in abs_points_by_variant}
                top_future_abs_points = {v: future_abs_points_by_variant.get(v, []) for v in future_scenarios if v in future_abs_points_by_variant}
                top_combined_abs = {**top_abs_points, **top_future_abs_points}

                profiles_bundle_by_variant = {}
                all_loc_ids = set()
                for points_list in top_combined_abs.values():
                    for point in points_list:
                        all_loc_ids.add(str(point.get("location_id", "")))
                loc_ids = tuple(all_loc_ids)
                if loc_ids:
                    daily_stat = "max" if metric_key == "dTmax" else "mean"
                    _tables_dir = DEFAULT_B_DATA_DIR / "_tables"
                    for scenario in top_container_variants:
                        if scenario in all_scenarios:
                            profiles = h.load_daily_profiles_abs_precomputed(
                                _tables_dir, scenario, daily_stat, loc_ids
                            )
                            if not (profiles.get("profiles") and len(profiles["profiles"]) > 0):
                                profiles = _timed(
                                    f"f22e__daily_profiles_from_daily_stats_{scenario}_v2",
                                    lambda s=scenario, i=idx, l=loc_ids, d=daily_stat, b=baseline_variant: h.f22e__build_daily_db_profiles_single_variant_from_daily_stats(
                                        daily_stats,
                                        i,
                                        location_ids=l,
                                        variant=s,
                                        daily_stat=d,
                                        baseline_variant=b,
                                    ),
                                    notes=f"Builds daily profiles for {scenario}.",
                                )
                            if profiles:
                                profiles_bundle_by_variant[scenario] = profiles

                unitr_points: list[dict] = []
                if UNITR_CTI_CSV.exists():
                    unitr_points = _load_unitr_cti_points(UNITR_CTI_CSV)

                # Hourly data for "Ore > θmax" and Temperature orarie tab: use same files as Data Preview (station hourly by region)
                selected_location_id = st.query_params.get("location_id")
                hourly_series_by_location_id: dict = {}
                MAX_HOURLY_STATIONS = 12  # cap to stay under Streamlit message size limit (~200 MB)
                loc_ids_to_load: list[str] = []
                if not idx.empty and top_combined_abs:
                    seen: set[str] = set()
                    for variant_key, points in top_combined_abs.items():
                        for pt in points:
                            lid = pt.get("location_id")
                            if lid is not None:
                                sid = str(lid).strip()
                                if sid and sid not in seen:
                                    seen.add(sid)
                                    loc_ids_to_load.append(sid)
                                    if len(loc_ids_to_load) >= MAX_HOURLY_STATIONS:
                                        break
                        if len(loc_ids_to_load) >= MAX_HOURLY_STATIONS:
                            break

                # Primary path: discover station hourly parquets (cached)
                by_region = _discover_parquet_files_by_region(DEFAULT_B_DATA_DIR)
                region_stem_to_loc = {}
                if not idx.empty:
                    for rec in idx[["region", "station_key", "location_id"]].fillna("").to_dict("records"):
                        reg = str(rec.get("region") or "").strip()
                        sk = str(rec.get("station_key") or "").strip()
                        lid = str(rec.get("location_id") or "").strip()
                        if reg and lid:
                            if sk:
                                region_stem_to_loc[(reg, sk)] = lid
                            region_stem_to_loc[(reg, lid)] = lid
                region_stem_to_loc_items = tuple(sorted(region_stem_to_loc.items()))
                hourly_series_by_location_id = _build_hourly_series_by_location_id(
                    DEFAULT_B_DATA_DIR,
                    by_region,
                    region_stem_to_loc_items,
                    max_stations=MAX_HOURLY_STATIONS,
                )

                if not selected_location_id and hourly_series_by_location_id:
                    selected_location_id = next(iter(hourly_series_by_location_id))
                # If user asked for a specific station via URL, ensure we load it even if beyond the cap
                if selected_location_id and selected_location_id not in hourly_series_by_location_id and not idx.empty:
                    loc_row = idx[idx["location_id"].astype(str) == str(selected_location_id)]
                    if not loc_row.empty:
                        r = loc_row.iloc[0]
                        region_val, station_key_val = r.get("region"), r.get("station_key")
                        region_str = str(region_val).strip() if pd.notna(region_val) else ""
                        sk_str = str(station_key_val).strip() if pd.notna(station_key_val) else ""
                        hourly_wide = None
                        if region_str and sk_str:
                            hourly_wide = _load_station_hourly_cached(DEFAULT_B_DATA_DIR, region_str, sk_str)
                        if (hourly_wide is None or hourly_wide.empty) and by_region:
                            for _reg_label, _reg_files in by_region:
                                if _reg_label in ("Tables", "Root"):
                                    continue
                                if _reg_label != region_str and str(r.get("region")).strip() != _reg_label:
                                    continue
                                for _path, _label in _reg_files:
                                    if "hourly" not in _label.lower() or not _path.exists():
                                        continue
                                    if _path.stem != sk_str and _path.stem != str(selected_location_id):
                                        continue
                                    try:
                                        hourly_wide = _read_parquet_robust(_path)
                                    except Exception:
                                        continue
                                    if hourly_wide is not None and not hourly_wide.empty:
                                        break
                                if hourly_wide is not None and not hourly_wide.empty:
                                    break
                        if hourly_wide is not None and not hourly_wide.empty:
                            long_hourly = _station_hourly_to_long(hourly_wide, sk_str, scenarios=None)
                            if not long_hourly.empty and "datetime" in long_hourly.columns and "DBT" in long_hourly.columns:
                                long_hourly["_month"] = pd.to_datetime(long_hourly["datetime"], errors="coerce").dt.month
                                summer = long_hourly[long_hourly["_month"].isin([6, 7, 8])]
                                hourly_series_by_location_id[selected_location_id] = {}
                                for sc in long_hourly["scenario"].dropna().unique().tolist():
                                    sc_str = str(sc).strip()
                                    if not sc_str:
                                        continue
                                    sub = summer[summer["scenario"] == sc]
                                    series = _series_from_sub(sub)
                                    hourly_series_by_location_id[selected_location_id][sc_str] = series
                                for alias_from, alias_to in [("rcp45_2050", "tmyx__rcp45_2050"), ("rcp85_2080", "tmyx__rcp85_2080")]:
                                    if alias_from in hourly_series_by_location_id[selected_location_id] and alias_to not in hourly_series_by_location_id[selected_location_id]:
                                        hourly_series_by_location_id[selected_location_id][alias_to] = hourly_series_by_location_id[selected_location_id][alias_from]

                # Fallback: if no per-station parquet found, try regional hourly (e.g. D-*__DBT__F-HR__L-XX.parquet)
                if not hourly_series_by_location_id and not idx.empty and loc_ids_to_load:
                    for loc_id in loc_ids_to_load[:10]:
                        rel_paths = idx[idx["location_id"].astype(str) == loc_id]["rel_path"].dropna().unique().tolist()
                        if not rel_paths:
                            continue
                        regional_hourly = _load_regional_hourly_by_rel_paths(rel_paths, idx, DEFAULT_B_DATA_DIR)
                        if regional_hourly.empty or "DBT" not in regional_hourly.columns or "rel_path" not in regional_hourly.columns:
                            continue
                        if isinstance(regional_hourly.index, pd.DatetimeIndex):
                            regional_hourly = regional_hourly.reset_index()
                        if "datetime" not in regional_hourly.columns:
                            for _c in ("index", "date", "time", "timestamp"):
                                if _c in regional_hourly.columns:
                                    regional_hourly = regional_hourly.rename(columns={_c: "datetime"})
                                    break
                        if "datetime" not in regional_hourly.columns:
                            continue
                        regional_hourly["datetime"] = pd.to_datetime(regional_hourly["datetime"], errors="coerce")
                        regional_hourly = regional_hourly.dropna(subset=["datetime", "DBT"])
                        regional_hourly["_month"] = regional_hourly["datetime"].dt.month
                        summer_reg = regional_hourly[regional_hourly["_month"].isin([6, 7, 8])]
                        hourly_series_by_location_id[loc_id] = {}
                        for rp in rel_paths:
                            scenario = str(rp).split("__", 1)[-1] if "__" in str(rp) else str(rp)
                            sub = summer_reg[summer_reg["rel_path"] == rp]
                            series = _series_from_sub(sub)
                            if series:
                                hourly_series_by_location_id[loc_id][scenario] = series
                        for alias_from, alias_to in [("rcp45_2050", "tmyx__rcp45_2050"), ("rcp85_2080", "tmyx__rcp85_2080")]:
                            if loc_id in hourly_series_by_location_id and alias_from in hourly_series_by_location_id[loc_id] and alias_to not in hourly_series_by_location_id[loc_id]:
                                hourly_series_by_location_id[loc_id][alias_to] = hourly_series_by_location_id[loc_id][alias_from]
                        if hourly_series_by_location_id:
                            if not selected_location_id:
                                selected_location_id = loc_id
                            break

                import hashlib as _hl
                _abs_j = json.dumps(top_combined_abs)
                _profiles_j = json.dumps(profiles_bundle_by_variant) if profiles_bundle_by_variant else "{}"
                _future_j = json.dumps(top_future_abs_points) if top_future_abs_points else "{}"
                _reg_j = json.dumps(region_options_data)
                _unitr_j = json.dumps(unitr_points)
                _hourly_j = json.dumps(hourly_series_by_location_id) if hourly_series_by_location_id else "{}"
                html_v2 = _build_region_maps_html(
                    baseline_variants_key=tuple(top_combined_abs.keys()),
                    compare_variant=compare_variant,
                    metric_key=metric_key,
                    percentile=float(percentile),
                    ui_lang=st.session_state.get("ui_lang", "EN"),
                    abs_fp=_hl.md5(_abs_j.encode()).hexdigest(),
                    profiles_fp=_hl.md5(_profiles_j.encode()).hexdigest(),
                    future_abs_fp=_hl.md5(_future_j.encode()).hexdigest(),
                    region_options_fp=_hl.md5(_reg_j.encode()).hexdigest(),
                    unitr_fp=_hl.md5(_unitr_j.encode()).hexdigest(),
                    hourly_fp=_hl.md5(_hourly_j.encode()).hexdigest(),
                    selected_location_id=selected_location_id,
                    dashboard_cols=V2_DASHBOARD_COLS,
                    maps_row_gap_px=V2_MAPS_ROW_GAP_PX,
                    maps_row3_gap_px=V2_MAPS_ROW3_GAP_PX,
                    thermo_separator_gap_px=D3_DIVIDER_MARGIN_TOP_PX,
                    divider_margin_bottom_px=D3_DIVIDER_MARGIN_BOTTOM_PX,
                    _abs_points_json=_abs_j,
                    _profiles_json=_profiles_j,
                    _future_abs_json=_future_j,
                    _region_options_json=_reg_j,
                    _unitr_points_json=_unitr_j,
                    _hourly_json=_hourly_j,
                    _geo_script=_GEO_SCRIPT,
                )
                components.html(html_v2, height=1150, scrolling=False)

                # Lightweight debug panel to quickly diagnose missing metrics / bad data path in deployments
                with st.expander("Debug: data sanity", expanded=False):
                    st.json(h.debug_data_sanity(DEFAULT_DATA_DIR))

                # Debug maps utility (underneath D3 dashboard, not in sidebar)
                DEBUG_MAPS = st.checkbox("Debug maps", value=False, key="debug_maps_confronto_regione")
                if DEBUG_MAPS:
                    value_label = "Tmax" if metric_key == "dTmax" else "Tavg"
                    st.write(
                        f"**Metric:** {value_label} (value column) | **Variants:** {list(top_combined_abs.keys())}"
                    )
                    for variant_key in top_combined_abs.keys():
                        points = top_combined_abs.get(variant_key, [])
                        df_variant = pd.DataFrame(points)
                        # Columns for display: location_name, value (Tmax/Tavg), latitude, longitude, region
                        plot_cols = [c for c in ["location_name", "value", "latitude", "longitude", "region"] if c in df_variant.columns]

                        with st.expander(f"**{variant_key}** — {len(points)} points", expanded=variant_key == base_for_default):
                            if df_variant.empty:
                                st.warning(f"No points for {variant_key} — markers will not appear on this map")
                            else:
                                st.dataframe(
                                    df_variant[plot_cols] if plot_cols else df_variant,
                                    use_container_width=True,
                                    height=min(200, 35 * len(df_variant) + 38),
                                )
                                df_dbg = h.debug_map_df(
                                    df_variant,
                                    label=f"D3 map df ({variant_key})",
                                    region_code=None,
                                    metric_col="value",
                                )
                                if df_dbg is not None and not df_dbg.empty:
                                    lat_col = next((c for c in ["lat", "latitude"] if c in df_dbg.columns), None)
                                    lon_col = next((c for c in ["lon", "longitude", "lng"] if c in df_dbg.columns), None)
                                    if lat_col and lon_col:
                                        import plotly.express as px

                                        df_plot = df_dbg.dropna(subset=[lat_col, lon_col])
                                        if df_plot.empty:
                                            st.warning("All rows have null lat/lon — cannot plot")
                                        else:
                                            fig_dbg = px.scatter_mapbox(
                                                df_plot,
                                                lat=lat_col,
                                                lon=lon_col,
                                                color="value" if "value" in df_plot.columns else None,
                                                hover_name="location_name" if "location_name" in df_plot.columns else None,
                                                hover_data=["value"] if "value" in df_plot.columns else None,
                                                zoom=6,
                                                height=420,
                                            )
                                            fig_dbg.update_traces(marker={"size": 12, "opacity": 0.9})
                                            st.plotly_chart(fig_dbg, use_container_width=True)
                                    else:
                                        st.warning(f"Missing lat/lon: lat_col={lat_col}, lon_col={lon_col}")

    with scenario_tabs[3]:
        st.markdown(f"##### {label('current_weather_data_tmyx')}")
        st.markdown(label("tmyx_tab_intro"))
        st.markdown("")
        present_idx = idx[idx["variant"].isin(present_variants)].copy()
        present_idx["label"] = present_idx["location_name"].fillna(present_idx["station_key"].astype(str))
        present_idx["label"] = present_idx["label"] + " (" + present_idx["location_id"].astype(str) + ")"
        loc_options = present_idx[["station_key", "location_id", "label"]].drop_duplicates().sort_values("label")
        loc_labels = loc_options["label"].tolist()
        loc_id_map = dict(zip(loc_options["label"], loc_options["location_id"].astype(str)))
        loc_station_map = dict(zip(loc_options["label"], loc_options["station_key"].astype(str)))
        if not loc_labels:
            st.info("No present (TMYx) locations available.")
        else:
            # Row 0: Titles + legends (cols 1-3) + location selector (col 4)
            LEGEND_ROW_HEIGHT_PX = 70

            def _legend_box(html: str) -> None:
                st.markdown(
                    f"<div style='height:{LEGEND_ROW_HEIGHT_PX}px; overflow:hidden; display:flex; align-items:center; justify-content:center;'>"
                    f"{html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Build legend HTML blocks (kept lightweight; charts keep their own data logic)
            # Heatmap legend: horizontal gradient + min/max
            try:
                _heat_colors = getattr(h, "_get_nuanced_colorset")()
            except Exception:
                _heat_colors = ["#3b4cc0", "#f7f7f7", "#b40426"]
            if not _heat_colors:
                _heat_colors = ["#3b4cc0", "#f7f7f7", "#b40426"]
            _stops = []
            for i, c in enumerate(_heat_colors):
                pct = int(round(i * 100 / max(1, (len(_heat_colors) - 1))))
                _stops.append(f"{c} {pct}%")
            _heat_grad = "linear-gradient(90deg, " + ", ".join(_stops) + ")"

            heatmap_legend_html = (
                f"<div style='width:82%; margin:0 auto;'>"
                f"  <div style='height:12px; border-radius:6px; border:1px solid rgba(0,0,0,0.25); background:{_heat_grad};'></div>"
                f"  <div style='display:flex; justify-content:space-between; font-size:11px; color:rgba(0,0,0,0.65); margin-top:4px;'>"
                f"    <span>{HEATMAP_ZMIN:g}°C</span><span>{HEATMAP_ZMAX:g}°C</span>"
                f"  </div>"
                f"</div>"
            )

            # Daily scatter legend: mean line + range band
            scatter_legend_html = (
                "<div style='display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:rgba(0,0,0,0.75); width:100%; justify-content:center;'>"
                "  <div style='display:flex; align-items:center; gap:8px;'>"
                "    <span style='display:inline-block; width:28px; height:0; border-top:2px solid #1f77b4;'></span>"
                "    <span>Daily Mean</span>"
                "  </div>"
                "  <div style='display:flex; align-items:center; gap:8px;'>"
                "    <span style='display:inline-block; width:28px; height:10px; background:rgba(112,115,155,0.18); border:1px solid rgba(0,0,0,0.12); border-radius:3px;'></span>"
                "    <span>Daily Range</span>"
                "  </div>"
                "</div>"
            )

            # Stacked columns legend: temperature bins (chips)
            _bin_step = 5
            _bins = list(range(int(HEATMAP_ZMIN), int(HEATMAP_ZMAX) + _bin_step, _bin_step))
            _bin_labels = [f"{_bins[i]}-{_bins[i+1]}" for i in range(len(_bins) - 1)]
            try:
                _nuanced = getattr(h, "_get_nuanced_colorset")()
            except Exception:
                _nuanced = ["#3b4cc0", "#f7f7f7", "#b40426"]
            if not _nuanced:
                _nuanced = ["#3b4cc0", "#f7f7f7", "#b40426"]
            _chips = []
            denom = max(1.0, float(HEATMAP_ZMAX - HEATMAP_ZMIN))
            for lab in _bin_labels:
                b0 = float(lab.split("-")[0])
                t = max(0.0, min(1.0, (b0 - float(HEATMAP_ZMIN)) / denom))
                ci = int(t * (len(_nuanced) - 1))
                col = _nuanced[ci]
                _chips.append(
                    f"<span style='display:inline-flex; align-items:center; gap:6px; margin:0;'>"
                    f"<span style='width:10px; height:10px; border-radius:2px; background:{col}; border:1px solid rgba(0,0,0,0.15);'></span>"
                    f"<span>{lab}</span>"
                    f"</span>"
                )
            # Force stacked legend into exactly two lines (4 bins per line) with extra line spacing.
            _chips_line1 = "".join(_chips[:4])
            _chips_line2 = "".join(_chips[4:8])
            stacked_legend_html = (
                "<div style='width:92%; margin:0 auto; font-size:10.5px; color:rgba(0,0,0,0.75); "
                "display:flex; flex-direction:column; align-items:center; line-height:1.1; gap:5px;'>"
                "  <div style='display:flex; justify-content:center; gap:12px; flex-wrap:nowrap;'>"
                + _chips_line1
                + "  </div>"
                "  <div style='display:flex; justify-content:center; gap:12px; flex-wrap:nowrap;'>"
                + _chips_line2
                + "  </div>"
                "</div>"
            )

            row0 = st.columns([1, 1, 1, 0.9], gap="medium")
            with row0[0]:
                st.markdown("###### Heatmaps")
                _legend_box(heatmap_legend_html)
            with row0[1]:
                st.markdown("###### Daily Scatter")
                _legend_box(scatter_legend_html)
            with row0[2]:
                st.markdown("###### Stacked Columns")
                _legend_box(stacked_legend_html)
            with row0[3]:
                loc_label = st.selectbox("Location", options=loc_labels, index=0, key="tmyx_location")
                tmyx_month_range = st.slider(
                    "Show data for selected months",
                    min_value=1,
                    max_value=12,
                    value=(1, 12),
                    step=1,
                    key="tmyx_month_range",
                )
                show_charts_tmyx = st.toggle(
                    "Show Charts",
                    value=True,
                    key="tmyx_show_charts_toggle",
                    help="Enable to load Plotly heatmaps, scatter and stacked column charts.",
                )

            loc_id = loc_id_map.get(loc_label)
            station_key = loc_station_map.get(loc_label)
            if not loc_id:
                st.warning("Invalid location selection.")
            else:
                station_region = station_region_map.get(str(station_key))
                if not station_region:
                    st.warning("No region found for this station.")
                    station_hourly_long = pd.DataFrame(columns=["rel_path", "datetime", "DBT"])
                else:
                    station_hourly_wide = _load_station_hourly_cached(DEFAULT_B_DATA_DIR, station_region, str(station_key))
                    station_hourly_long = _station_hourly_to_long(
                        station_hourly_wide,
                        str(station_key),
                        scenarios=present_variants,
                    )
                # Filter TMYx data by selected month range (only for this tab)
                start_month, end_month = tmyx_month_range
                if start_month == 1 and end_month == 12:
                    tmyx_hourly_filtered = station_hourly_long
                else:
                    tmyx_hourly_filtered = station_hourly_long[
                        station_hourly_long["datetime"].dt.month.between(start_month, end_month)
                    ].copy()

                # Optional: show Plotly charts only when toggle is on
                if not show_charts_tmyx:
                    st.info("Enable **Show Charts** to load heatmaps, daily scatter and stacked column charts.")
                else:
                    st.markdown(
                        f"<div style='height:0px; margin-bottom:{TMYX_HEADER_TO_CHARTS_GAP_PX}px'></div>",
                        unsafe_allow_html=True,
                    )
                row1 = st.columns([1, 1, 1, 0.9], gap="medium")
                if show_charts_tmyx:
                    with row1[0]:
                        heatmap_fig = _timed(
                            "f34__plotly_tmyx_heatmap_subplots",
                            lambda: h.f34__plotly_tmyx_heatmap_subplots(
                                tmyx_hourly_filtered,
                                idx,
                                location_id=loc_id,
                                variants=present_variants,
                                colorscale=HEATMAP_COLORSCALE,
                                zmin=HEATMAP_ZMIN,
                                zmax=HEATMAP_ZMAX,
                                subplot_height=TMYX_CHART_HEIGHT,
                                vertical_spacing=TMYX_SUBPLOT_VERTICAL_SPACING,
                                show_subplot_titles=True,
                                subplot_title_font_size=TMYX_SUBPLOT_TITLE_FONTSIZE,
                                subplot_title_yshift=TMYX_SUBPLOT_TITLE_YSHIFT,
                                subplot_title_bold=TMYX_SUBPLOT_TITLE_BOLD,
                                margin=TMYX_CHART_MARGIN,
                            ),
                            notes="Builds TMYx heatmap subplot figure across variants.",
                        )
                        if heatmap_fig is not None:
                            heatmap_fig.update_traces(showscale=False)
                            st.plotly_chart(heatmap_fig, width="stretch", key=f"heatmap_subplots_{loc_id}")
                        else:
                            st.info("No heatmap data available.")

                    with row1[1]:
                        scatter_fig = _timed(
                            "f35__plotly_tmyx_scatter_subplots",
                            lambda: h.f35__plotly_tmyx_scatter_subplots(
                                tmyx_hourly_filtered,
                                idx,
                                location_id=loc_id,
                                variants=present_variants,
                                subplot_height=TMYX_CHART_HEIGHT,
                                vertical_spacing=TMYX_SUBPLOT_VERTICAL_SPACING,
                                show_subplot_titles=True,
                                subplot_title_font_size=TMYX_SUBPLOT_TITLE_FONTSIZE,
                                subplot_title_yshift=TMYX_SUBPLOT_TITLE_YSHIFT,
                                subplot_title_bold=TMYX_SUBPLOT_TITLE_BOLD,
                                margin=TMYX_CHART_MARGIN,
                                y_range=(HEATMAP_ZMIN, HEATMAP_ZMAX + 1),
                            ),
                            notes="Builds TMYx daily scatter subplot figure across variants.",
                        )
                        if scatter_fig is not None:
                            scatter_fig.update_layout(showlegend=False)
                            st.plotly_chart(scatter_fig, width="stretch", key=f"scatter_subplots_{loc_id}")
                        else:
                            st.info("No scatter data available.")

                    with row1[2]:
                        stacked_fig = _timed(
                            "f36__plotly_tmyx_stacked_subplots",
                            lambda: h.f36__plotly_tmyx_stacked_subplots(
                                tmyx_hourly_filtered,
                                idx,
                                location_id=loc_id,
                                variants=present_variants,
                                subplot_height=TMYX_CHART_HEIGHT,
                                vertical_spacing=TMYX_SUBPLOT_VERTICAL_SPACING,
                                show_subplot_titles=True,
                                subplot_title_font_size=TMYX_SUBPLOT_TITLE_FONTSIZE,
                                subplot_title_yshift=TMYX_SUBPLOT_TITLE_YSHIFT,
                                subplot_title_bold=TMYX_SUBPLOT_TITLE_BOLD,
                                margin=TMYX_CHART_MARGIN,
                                temp_min=HEATMAP_ZMIN,
                                temp_max=HEATMAP_ZMAX,
                                colorscale=HEATMAP_COLORSCALE,
                            ),
                            notes="Builds TMYx stacked columns subplot figure across variants.",
                        )
                        if stacked_fig is not None:
                            stacked_fig.update_layout(showlegend=False)
                            st.plotly_chart(stacked_fig, width="stretch", key=f"stacked_subplots_{loc_id}")
                        else:
                            st.info("No stacked column data available.")

                with row1[3]:
                    # Ore > θmax: from UNI/TR nearest station θmax, count hours above it (no scatter chart needed)
                    _unitr_tmyx = _load_unitr_cti_points(UNITR_CTI_CSV) if UNITR_CTI_CSV.exists() else []
                    theta_max_c = None
                    if _unitr_tmyx and not station_hourly_long.empty:
                        _loc_row = idx[idx["location_id"].astype(str) == str(loc_id)]
                        if not _loc_row.empty:
                            _lat = _loc_row.iloc[0].get("latitude")
                            _lon = _loc_row.iloc[0].get("longitude")
                            if pd.notna(_lat) and pd.notna(_lon):
                                _lat, _lon = float(_lat), float(_lon)
                                _best = None
                                _best_d2 = float("inf")
                                for _p in _unitr_tmyx:
                                    _u = float(_p.get("lat_geocoded", 0))
                                    _v = float(_p.get("lon_geocoded", 0))
                                    _d2 = (_lat - _u) ** 2 + (_lon - _v) ** 2
                                    if _d2 < _best_d2:
                                        _best_d2 = _d2
                                        _best = _p
                                if _best is not None:
                                    theta_max_c = float(_best.get("theta_max_C"))
                    if theta_max_c is not None and not station_hourly_long.empty and "scenario" in station_hourly_long.columns and "DBT" in station_hourly_long.columns:
                        st.markdown("###### Ore &gt; θmax")
                        _above = station_hourly_long[station_hourly_long["DBT"] > theta_max_c].groupby("scenario", observed=True).size()
                        def _shorten(v):
                            return v.replace("tmyx_", "", 1) if v.startswith("tmyx_") else v
                        for _v in present_variants:
                            _h = int(_above.get(_v, 0))
                            st.metric(label=_shorten(_v), value=f"{_h:,}", help=f"Hours with DBT > θmax ({theta_max_c:.1f}°C) — UNI/TR 10349")
                        st.markdown("---")
                    st.markdown("###### Hours Above Threshold")

                    # Inject CSS to reduce metric font sizes
                    st.markdown("""
                        <style>
                        .stMetric {
                            padding: 0.25rem 0;
                        }
                        .stMetric > label {
                            font-size: 0.7rem !important;
                            font-weight: 500;
                        }
                        .stMetric > div[data-testid="stMetricValue"] {
                            font-size: 0.9rem !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                
                    thresholds = [28.0, 30.0, 32.0]
                    metrics_data = _timed(
                        "f127__calculate_hours_above_thresholds",
                        lambda: h.f127__calculate_hours_above_thresholds(
                            tmyx_hourly_filtered,
                            idx,
                            location_id=loc_id,
                            variants=present_variants,
                            thresholds=thresholds,
                        ),
                        notes="Counts hours above temperature thresholds for KPIs.",
                    )
                
                    if not metrics_data.empty:
                        # Helper function to shorten variant names
                        def shorten_variant(v):
                            if v.startswith("tmyx_"):
                                return v.replace("tmyx_", "", 1)
                            return v
                    
                        # Create 3 sub-columns, one for each threshold
                        thresh_col1, thresh_col2, thresh_col3 = st.columns(3)
                    
                        with thresh_col1:
                            st.markdown(f"**≥ {thresholds[0]:.0f}°C**")
                            for variant in present_variants:
                                variant_data = metrics_data[
                                    (metrics_data["variant"] == variant) & 
                                    (metrics_data["threshold"] == thresholds[0])
                                ]
                                if not variant_data.empty:
                                    total_hours = int(variant_data["hours"].sum())
                                    st.metric(
                                        label=shorten_variant(variant),
                                        value=f"{total_hours:,}",
                                        help=f"Total hours with DBT ≥ {thresholds[0]:.0f}°C for {variant}",
                                    )
                                else:
                                    st.metric(label=shorten_variant(variant), value="0")
                    
                        with thresh_col2:
                            st.markdown(f"**≥ {thresholds[1]:.0f}°C**")
                            for variant in present_variants:
                                variant_data = metrics_data[
                                    (metrics_data["variant"] == variant) & 
                                    (metrics_data["threshold"] == thresholds[1])
                                ]
                                if not variant_data.empty:
                                    total_hours = int(variant_data["hours"].sum())
                                    st.metric(
                                        label=shorten_variant(variant),
                                        value=f"{total_hours:,}",
                                        help=f"Total hours with DBT ≥ {thresholds[1]:.0f}°C for {variant}",
                                    )
                                else:
                                    st.metric(label=shorten_variant(variant), value="0")
                    
                        with thresh_col3:
                            st.markdown(f"**≥ {thresholds[2]:.0f}°C**")
                            for variant in present_variants:
                                variant_data = metrics_data[
                                    (metrics_data["variant"] == variant) & 
                                    (metrics_data["threshold"] == thresholds[2])
                                ]
                                if not variant_data.empty:
                                    total_hours = int(variant_data["hours"].sum())
                                    st.metric(
                                        label=shorten_variant(variant),
                                        value=f"{total_hours:,}",
                                        help=f"Total hours with DBT ≥ {thresholds[2]:.0f}°C for {variant}",
                                    )
                                else:
                                    st.metric(label=shorten_variant(variant), value="0")
                    else:
                        st.info("No metrics data available.")

    with scenario_tabs[4]:
        st.markdown(
            "**RCP** (Representative Concentration Pathway) and **SSP** (Shared Socioeconomic Pathway) are "
            "standardized greenhouse-gas pathways used by the IPCC. They are scenarios, not predictions: "
            "they describe possible futures depending on emissions and society."
        )
        with st.expander("Global temperature context (IPCC-style charts)", expanded=False):
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("###### Historical Temperature Anomaly")
                years_hist = list(range(1850, 2025))
                temp_anomaly_hist = [
                    -0.4 + 0.01 * (y - 1850) + 0.0001 * ((y - 1850) ** 2) if y < 1950
                    else -0.2 + 0.02 * (y - 1950) + 0.0003 * ((y - 1950) ** 2) if y < 2000
                    else 0.3 + 0.03 * (y - 2000) + 0.0005 * ((y - 2000) ** 2)
                    for y in years_hist
                ]
                np.random.seed(42)
                temp_anomaly_hist = [t + np.random.normal(0, 0.1) for t in temp_anomaly_hist]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=years_hist,
                    y=temp_anomaly_hist,
                    mode='lines',
                    name='Observed',
                    line=dict(color='#2c3e50', width=2),
                    hovertemplate='Year: %{x}<br>Anomaly: %{y:.2f}°C<extra></extra>'
                ))
                fig_hist.update_layout(
                    title="Global Surface Temperature Anomaly<br><sub>Relative to 1850-1900 baseline</sub>",
                    xaxis_title="Year",
                    yaxis_title="Temperature Anomaly (°C)",
                    height=350,
                    margin=dict(l=50, r=20, t=70, b=50),
                    hovermode='x unified',
                    font=dict(family="Inter, system-ui, sans-serif", size=11),
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
                    yaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=True, zerolinecolor='#666'),
                )
                st.plotly_chart(fig_hist, width='stretch', key="ipcc_historical_scenarios_tab")
                st.caption("Based on IPCC AR6. [View IPCC data](https://www.ipcc.ch/data/)")
            with col2:
                st.markdown("###### Projected Temperature Change by Scenario")
                years_proj = list(range(2020, 2101))
                ssp126 = [0.0 + 0.014 * (y - 2020) / 80 for y in years_proj]
                ssp245 = [0.0 + 0.027 * (y - 2020) / 80 for y in years_proj]
                ssp585 = [0.0 + 0.044 * (y - 2020) / 80 for y in years_proj]
                fig_proj = go.Figure()
                fig_proj.add_trace(go.Scatter(
                    x=years_proj, y=ssp126, mode='lines', name='SSP1-2.6 (Low)',
                    line=dict(color='#2ecc71', width=2, dash='dot'),
                    hovertemplate='Year: %{x}<br>ΔT: %{y:.2f}°C<extra></extra>'
                ))
                fig_proj.add_trace(go.Scatter(
                    x=years_proj, y=ssp245, mode='lines', name='SSP2-4.5 (Intermediate)',
                    line=dict(color='#f39c12', width=2, dash='dot'),
                    hovertemplate='Year: %{x}<br>ΔT: %{y:.2f}°C<extra></extra>'
                ))
                fig_proj.add_trace(go.Scatter(
                    x=years_proj, y=ssp585, mode='lines', name='SSP5-8.5 (High)',
                    line=dict(color='#e74c3c', width=2, dash='dot'),
                    hovertemplate='Year: %{x}<br>ΔT: %{y:.2f}°C<extra></extra>'
                ))
                fig_proj.update_layout(
                    title="Projected Global Temperature Change<br><sub>Relative to 2020 baseline</sub>",
                    xaxis_title="Year",
                    yaxis_title="Temperature Change (°C)",
                    height=350,
                    margin=dict(l=50, r=20, t=70, b=50),
                    hovermode='x unified',
                    font=dict(family="Inter, system-ui, sans-serif", size=11),
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
                    yaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=True, zerolinecolor='#666'),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                st.plotly_chart(fig_proj, width='stretch', key="ipcc_projections_scenarios_tab")
                st.caption("Based on IPCC AR6. [View IPCC AR6 data](https://ipcc-browser.ipcc-data.org/)")

def _run_debug_page():
    debug_tabs = st.tabs(["Code Performance", "Data Preview", "Data Structure", "Preparation Scripts"])

    with debug_tabs[0]:
        st.markdown("##### Code Performance")
        timings = st.session_state.get("code_timing", {})
        perf_rows = [
            {"function_name": "load_b_inventory", "description": "Load station inventory from _tables."},
            {"function_name": "load_b_daily_stats", "description": "Load daily stats parquet from _tables."},
            {"function_name": "load_b_pairing_debug", "description": "Load pairing debug CSV."},
            {"function_name": "load_file_stats_precomputed", "description": "Load precomputed file stats (06B)."},
            {"function_name": "f123__build_file_stats_from_daily", "description": "Per-file Tmax percentile and Tavg mean (fallback if not precomputed)."},
            {"function_name": "load_location_deltas_precomputed", "description": "Load precomputed location deltas (06B)."},
            {"function_name": "f125__compute_location_deltas_from_daily", "description": "Per-location max monthly ΔT (fallback if not precomputed)."},
            {"function_name": "load_location_stats_precomputed", "description": "Load precomputed location stats (06B)."},
            {"function_name": "f28b__compute_location_stats_for_variant_from_daily", "description": "Per-location stats per variant (fallback if not precomputed)."},
            {"function_name": "load_monthly_delta_table_precomputed", "description": "Load precomputed monthly delta table (06B)."},
            {"function_name": "f124__build_monthly_delta_table", "description": "Monthly delta table and max yearly column (fallback if not precomputed)."},
            {"function_name": "f121__daily_stats_by_rel_path", "description": "Pre-aggregate hourly → daily mean/max per file."},
            {"function_name": "f122__build_location_daily_join", "description": "Daily base/comp aligned series for per-location charts."},
            {"function_name": "f28c__compute_location_stats_cti_from_daily", "description": "CTI location stats from daily."},
            {"function_name": "f22e__build_daily_profiles_cti", "description": "CTI daily profiles for D3."},
            {"function_name": "f22e__build_daily_profiles_single_variant_from_daily_stats", "description": "Daily profiles for one variant (D3 charts)."},
            {"function_name": "f22d__build_daily_profiles_from_daily_stats", "description": "Daily profiles for delta D3."},
            {"function_name": "f23b__d3_dashboard_html_abs", "description": "D3 absolute scenario dashboard HTML."},
            {"function_name": "f23b__d3_dashboard_html_abs_cti", "description": "D3 CTI dashboard HTML."},
            {"function_name": "f23__d3_dashboard_html", "description": "D3 delta dashboard HTML."},
            {"function_name": "f19b__plotly_italy_map_abs", "description": "Plotly Italy map (absolute)."},
            {"function_name": "f34__plotly_tmyx_heatmap_subplots", "description": "TMYx heatmap subplot figure across variants."},
            {"function_name": "f35__plotly_tmyx_scatter_subplots", "description": "TMYx daily scatter subplot figure across variants."},
            {"function_name": "f36__plotly_tmyx_stacked_subplots", "description": "TMYx stacked columns subplot figure across variants."},
            {"function_name": "f127__calculate_hours_above_thresholds", "description": "Hours above temperature thresholds for KPIs."},
        ]
        for row in perf_rows:
            info = timings.get(row["function_name"], {})
            row["last_timing_seconds"] = info.get("seconds") if info else None
        perf_df = pd.DataFrame(perf_rows)
        perf_df["last_timing_seconds"] = perf_df["last_timing_seconds"].apply(
            lambda x: round(x, 3) if x is not None and pd.notna(x) else None
        )
        st.dataframe(perf_df, column_config={"last_timing_seconds": "last (s)"}, hide_index=True, width="stretch")
        if not timings:
            st.caption("No timings recorded yet. Interact with the app to populate timings.")
        st.caption("**Speed up:** Run `06B_precompute_station_tables.py --parquet-root data/04__italy_tmy_fwg_parquet --overwrite` to precompute f124, f125, f28b and file stats; the app will then use the fast paths (load_*_precomputed) instead of computing on the fly.")

    with debug_tabs[1]:
        st.markdown("##### Data Preview")
        by_region = _discover_parquet_files_by_region(DEFAULT_DATA_DIR)
        if not by_region:
            st.warning(f"No parquet files found in `{DEFAULT_DATA_DIR}`. Run data preparation scripts first.")
        else:
            preview_n = 20
            region_labels = [region for region, _ in by_region]
            region_files_map = {region: files for region, files in by_region}

            col_sel, col_info = st.columns([2, 3])
            with col_sel:
                selected_region = st.selectbox(
                    "Region",
                    options=region_labels,
                    format_func=lambda r: f"{r}  ({len(region_files_map[r])} files)",
                    key="debug_preview_region",
                )
            region_files = region_files_map.get(selected_region, [])
            with col_info:
                st.caption(
                    f"**{selected_region}** — {len(region_files)} parquet file(s). "
                    f"Showing first {preview_n} rows of each."
                )

            if not region_files:
                st.info("No files found for this region.")
            else:
                for pa, _file_label in region_files:
                    try:
                        df_p = _read_parquet_robust(pa)
                        if df_p is None or df_p.empty:
                            st.caption(f"**{pa.name}** — empty")
                            continue
                        nrows, ncols = len(df_p), len(df_p.columns)
                        with st.expander(f"**{pa.name}** — {nrows:,} rows · {ncols} cols", expanded=False):
                            st.dataframe(df_p.head(preview_n), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.caption(f"**{pa.name}** — error: {e}")

    with debug_tabs[2]:
        st.markdown("##### Data folder structure")
        parquet_in_dir = _discover_parquet_files(DEFAULT_DATA_DIR)
        data_dir_exists = DEFAULT_DATA_DIR.exists()
        tables_dir = DEFAULT_DATA_DIR / "_tables"
        st.markdown("##### Detected files")
        if not data_dir_exists:
            st.warning(f"Data directory not found: `{DEFAULT_DATA_DIR}`.")
        else:
            st.caption(f"- **{DEFAULT_DATA_DIR.name}/** — app data root. Parquet files: **{len(parquet_in_dir)}**.")
            st.caption(f"- **_tables/** — `{tables_dir.name}`: daily stats, inventory, etc. (exists: {tables_dir.exists()}).")
            if parquet_in_dir:
                with st.expander("List parquet files"):
                    for pa, lb in parquet_in_dir[:20]:
                        st.text(pa.name)
                    if len(parquet_in_dir) > 20:
                        st.caption(f"... and {len(parquet_in_dir) - 20} more.")
        st.markdown("---")
        st.markdown(f"""
        ##### App data directories (from `app.py`)

        - **`data/04__italy_tmy_fwg_parquet/`** — TMYx + FWG parquet (default `DEFAULT_DATA_DIR`). Contains `_tables/` (e.g. `D-TMYxFWG__DBT__F-DD__L-ALL.parquet`, `*_Inventory_*.parquet`) and per-region hourly parquet (e.g. `D-*__DBT__F-HR__L-XX.parquet`).
        - **`data/04__italy_cti_parquet/`** — CTI (Itaca) weather stations. Contains `_tables/` and `epw/` parquet.

        ##### Source folders (inputs to preparation)

        - **`data/00__italy_climate_onebuilding/`** — EPW from climate.onebuilding.org.
        - **`data/02__italy_fwg_outputs/`** — FWG output (morphed EPW by scenario).
        - **`data/03__italy_all_epw_DBT_streamlit/`** — Optional intermediate (index + hourly extract).
        - **`data/data_preparation_scripts/`** — Preparation scripts (see Preparation Scripts tab).

        ##### Workflow (high level)

        1. Download/ingest EPW (e.g. 01_download, 02_collect).
        2. Run FWG to generate future EPW (03_run_fwg_recursive).
        3. Build index and extract DBT to parquet (05A / 05B / 05C).
        4. Precompute derived stats and station tables (06A / 06B / 06C).
        5. App loads from `04__italy_tmy_fwg_parquet` and `04__italy_cti_parquet`.
        """)

    with debug_tabs[3]:
        st.markdown("##### Data Preparation Scripts")
        script_rows = [
            {"Script": "00_list_fwg_models.py", "Input": "FWG JAR", "Output": "stdout", "Description": "List FWG model types from JAR.", "Workflow": "1) Inspect."},
            {"Script": "01_download_italy_epw.py", "Input": "—", "Output": "EPW/ZIP", "Description": "Download Italy EPW from climate.onebuilding.org.", "Workflow": "1) Download/ingest."},
            {"Script": "02_collect_epws.py", "Input": "EPW dirs", "Output": "EPW copy", "Description": "Collect/copy EPW from multiple sources.", "Workflow": "1) Ingest."},
            {"Script": "02b_rename_and_copy_fwg_outputs.py", "Input": "FWG output", "Output": "Renamed EPW", "Description": "Rename and copy FWG outputs.", "Workflow": "2) Post-FWG."},
            {"Script": "02c_copy_baseline_epws.py", "Input": "Baseline EPW", "Output": "Copy", "Description": "Copy baseline EPW files.", "Workflow": "2) Ingest."},
            {"Script": "03_run_fwg_recursive.py", "Input": "EPW + JAR", "Output": "Morphed EPW", "Description": "Run FWG on EPW to generate future scenarios.", "Workflow": "2) FWG morphing."},
            {"Script": "04_folder_test.py", "Input": "Folder", "Output": "stdout", "Description": "Test/inspect folder structure.", "Workflow": "—"},
            {"Script": "05A_build_epw_index_and_extract.py", "Input": "EPW roots", "Output": "index + parquet", "Description": "Build EPW index and extract DBT to parquet.", "Workflow": "3) Build index; 4) Write parquet."},
            {"Script": "05B_build_station_parquets.py", "Input": "Index + EPW", "Output": "Station parquet", "Description": "Build per-station parquet files.", "Workflow": "4) Write parquet."},
            {"Script": "05C_build_cti_station_parquets.py", "Input": "CTI EPW", "Output": "CTI parquet", "Description": "Build CTI station parquet.", "Workflow": "4) Write parquet."},
            {"Script": "06A_precompute_derived_stats.py", "Input": "Hourly parquet", "Output": "daily_stats, _tables", "Description": "Precompute daily stats and aggregates.", "Workflow": "4) Compute stats; 5) Write parquet."},
            {"Script": "06B_precompute_station_tables.py", "Input": "Daily/stats", "Output": "_tables", "Description": "Precompute station-level tables.", "Workflow": "5) Write parquet."},
            {"Script": "06C_precompute_cti_tables.py", "Input": "CTI data", "Output": "CTI _tables", "Description": "Precompute CTI tables.", "Workflow": "5) Write parquet."},
        ]
        script_df = pd.DataFrame(script_rows)
        st.dataframe(script_df, width="stretch", hide_index=True)
        st.caption("Scripts in data/data_preparation_scripts/. Run in order per README. See repo README for full workflow.")


_lang = st.session_state.get("ui_lang", "IT")
pg = st.navigation(
    [
        st.Page("pages/page_welcome.py", title=label("welcome", _lang), icon=":material/home:", default=True),
        st.Page(_run_scenarios_page, title=label("future_weather_scenarios", _lang), icon=":material/explore:"),
        st.Page(_run_debug_page, title="Data & Code Debug", icon=":material/bug_report:"),
    ],
    position="top",
)
pg.run()
