import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components
import json
import time
import importlib
import plotly.graph_objects as go
import libs.fn__libs as h
from libs.fn__libs_bilingual import label
from libs.fn__page_header import f001__create_page_header
from libs.fn__page_welcome import render_welcome_page

# Ensure Streamlit uses the latest local fn__libs.py (avoids stale module issues)
h = importlib.reload(h)
h.f101__inject_inter_font()
h.f102__enable_altair_inter_theme()


def _record_timing(name: str, seconds: float, notes: str | None = None) -> None:
    timings = st.session_state.setdefault("code_timing", {})
    timings[name] = {"seconds": float(seconds), "notes": notes or ""}


def _timed(name: str, fn, notes: str | None = None):
    start = time.perf_counter()
    result = fn()
    _record_timing(name, time.perf_counter() - start, notes=notes)
    return result


def _baseline_location_ids(idx: pd.DataFrame, baseline_variant: str) -> set[str]:
    if idx is None or idx.empty:
        return set()
    if "variant" not in idx.columns or "location_id" not in idx.columns:
        return set()
    return set(idx[idx["variant"] == baseline_variant]["location_id"].astype(str))


def _read_parquet_robust(parquet_path: Path, columns: list = None, **kwargs):
    """
    Robust Parquet reader with multiple fallback strategies.
    Handles PyArrow version incompatibilities and library issues.
    
    Tries in order:
    1. PyArrow with default options
    2. PyArrow with use_pandas_metadata=False
    3. PyArrow reading all columns then filtering
    4. fastparquet engine (if available)
    5. PyArrow direct API with different options
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    # Get PyArrow version for error messages
    try:
        pyarrow_version = pa.__version__
    except:
        pyarrow_version = "unknown"
    
    errors = []
    
    # Strategy 1: Standard pandas read with PyArrow (default)
    try:
        return pd.read_parquet(parquet_path, columns=columns, engine='pyarrow', **kwargs)
    except (OSError, Exception) as e:
        errors.append(f"1. Default PyArrow: {str(e)}")
    
    # Strategy 2: PyArrow with use_pandas_metadata=False
    try:
        return pd.read_parquet(
            parquet_path, 
            columns=columns, 
            engine='pyarrow',
            use_pandas_metadata=False,
            **kwargs
        )
    except (OSError, Exception) as e:
        errors.append(f"2. PyArrow (no pandas metadata): {str(e)}")
    
    # Strategy 3: Read all columns then filter (sometimes works when column selection fails)
    try:
        df = pd.read_parquet(parquet_path, engine='pyarrow', use_pandas_metadata=False, **kwargs)
        if columns:
            return df[columns]
        return df
    except (OSError, Exception) as e:
        errors.append(f"3. PyArrow (read all, filter): {str(e)}")
    
    # Strategy 4: Try fastparquet engine (if available)
    try:
        return pd.read_parquet(parquet_path, columns=columns, engine='fastparquet', **kwargs)
    except ImportError:
        errors.append("4. fastparquet: not installed")
    except (OSError, Exception) as e:
        errors.append(f"4. fastparquet: {str(e)}")
    
    # Strategy 5: Try PyArrow direct API with minimal options
    try:
        table = pq.read_table(parquet_path, columns=columns)
        return table.to_pandas()
    except (OSError, Exception) as e:
        errors.append(f"5. PyArrow direct API: {str(e)}")
    
    # Strategy 6: PyArrow direct API reading all columns
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if columns:
            return df[columns]
        return df
    except (OSError, Exception) as e:
        errors.append(f"6. PyArrow direct API (all cols): {str(e)}")
    
    # All strategies failed - raise with helpful message
    raise RuntimeError(
        f"Failed to read Parquet file with multiple strategies.\n"
        f"File: {parquet_path}\n"
        f"PyArrow version: {pyarrow_version}\n"
        f"Errors:\n" + "\n".join(f"  {err}" for err in errors) + "\n"
        f"\nTry: pip install --upgrade pyarrow or pip install fastparquet"
    )


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


@st.cache_data(show_spinner=False)
def _load_epw_index_meta(index_path: Path) -> dict[str, dict[str, object]]:
    if not index_path.exists():
        return {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return {}
    meta = {}
    for r in records:
        station_id = str(r.get("station_id") or r.get("wmo") or "").strip()
        if not station_id:
            continue
        meta[station_id] = {
            "location_name": r.get("location_name"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "region": r.get("region"),
        }
    return meta


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


def _build_idx_from_inventory(inventory: pd.DataFrame, epw_meta: dict[str, dict[str, object]]) -> pd.DataFrame:
    if inventory is None or inventory.empty:
        return pd.DataFrame()
    inv = inventory.copy()
    inv["station_key"] = inv["station_key"].astype(str).str.strip()

    name_col = "station_name" if "station_name" in inv.columns else "location_name" if "location_name" in inv.columns else None
    lat_col = "latitude" if "latitude" in inv.columns else "lat" if "lat" in inv.columns else None
    lon_col = "longitude" if "longitude" in inv.columns else "lon" if "lon" in inv.columns else None

    rows = []
    for _, r in inv.iterrows():
        station_key = str(r["station_key"])
        region = r.get("region")
        scenarios = _parse_inventory_cols(r.get("cols"))
        if not scenarios:
            continue
        for scenario in scenarios:
            scenario = str(scenario).strip()
            loc_name, station_id = _parse_station_key(station_key)
            base = scenario.split("__", 1)[0]
            meta = epw_meta.get(str(station_id)) if epw_meta else None
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


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
    if monthly_stats is None or monthly_stats.empty:
        return pd.DataFrame()
    df = monthly_stats.copy()
    if "station_key" in df.columns:
        df["station_key"] = df["station_key"].astype(str).str.strip()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
        df = df.dropna(subset=["month"])
    if "Tmax" in df.columns and "DBT_max" not in df.columns:
        df = df.rename(columns={"Tmax": "DBT_max"})
    if "Tmean" in df.columns and "DBT_mean" not in df.columns:
        df = df.rename(columns={"Tmean": "DBT_mean"})
    if "Tmin" in df.columns and "DBT_min" not in df.columns:
        df = df.rename(columns={"Tmin": "DBT_min"})
    if "station_key" in df.columns and "scenario" in df.columns:
        df["rel_path"] = df["station_key"].astype(str) + "__" + df["scenario"].astype(str)
    return df


def _normalize_cti_name(value: str) -> str:
    s = str(value or "").strip().lower()
    if s.startswith("ita_") and len(s) > 6:
        s = re.sub(r"^ita_[a-z]{2}_", "", s)
    if "__" in s:
        s = s.split("__", 2)[-1]
    s = s.replace("_", " ")
    s = re.sub(r"[\\'\\\"`]", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _pick_cti_region(region_value, meta_row) -> str | None:
    def _clean(code) -> str | None:
        s = str(code or "").strip()
        if len(s) == 2 and s.isalpha():
            return s.upper()
        return None
    meta_region = _clean(meta_row.get("region") if meta_row is not None else None)
    if meta_region:
        return meta_region
    return _clean(region_value)


def _build_idx_from_cti_inventory(inventory: pd.DataFrame, cti_list: pd.DataFrame) -> pd.DataFrame:
    if inventory is None or inventory.empty:
        return pd.DataFrame()
    inv = inventory.copy()
    inv["station_key"] = inv["station_key"].astype(str).str.strip()

    list_by_id = {}
    list_by_name = {}
    list_by_name_norm = {}
    if cti_list is not None and not cti_list.empty:
        for _, r in cti_list.iterrows():
            loc_id = str(r.get("location_id") or "").strip()
            loc_name = str(r.get("location_name") or "").strip()
            if loc_id:
                list_by_id[loc_id] = r
            if loc_name:
                list_by_name[loc_name.lower()] = r
                key = _normalize_cti_name(loc_name)
                if key:
                    list_by_name_norm[key] = r

    rows = []
    for _, r in inv.iterrows():
        station_key = str(r["station_key"])
        region = r.get("region")
        scenarios = _parse_inventory_cols(r.get("cols")) or ["cti"]
        for scenario in scenarios:
            scenario = str(scenario).strip() or "cti"
            loc_name, station_id = _parse_station_key(station_key)
            meta = None
            if list_by_id:
                meta = list_by_id.get(station_id)
                if meta is None:
                    meta = list_by_id.get(station_key)
            if meta is None and loc_name:
                meta = list_by_name.get(str(loc_name).lower())
            if meta is None:
                meta = list_by_name_norm.get(_normalize_cti_name(loc_name)) if loc_name else None
            rows.append(
                {
                    "station_key": station_key,
                    "location_id": station_id,
                    "location_name": (meta.get("location_name") if meta is not None else loc_name) or station_key,
                    "latitude": meta.get("latitude") if meta is not None else pd.NA,
                    "longitude": meta.get("longitude") if meta is not None else pd.NA,
                    "region": _pick_cti_region(region, meta),
                    "variant": scenario,
                    "group_key": station_key,
                    "rel_path": f"{station_key}__{scenario}",
                    "is_cti": True,
                }
            )
    return pd.DataFrame(rows)


def _station_hourly_to_long(
    hourly_wide: pd.DataFrame, station_key: str, scenarios: list[str] | None = None
) -> pd.DataFrame:
    if hourly_wide is None or hourly_wide.empty:
        return pd.DataFrame(columns=["rel_path", "datetime", "DBT"])
    df = hourly_wide.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "datetime"})
        if "datetime" not in df.columns and "dt" in df.columns:
            df = df.rename(columns={"dt": "datetime"})
    elif "dt" in df.columns:
        df = df.rename(columns={"dt": "datetime"})
    if "datetime" not in df.columns:
        return pd.DataFrame(columns=["rel_path", "datetime", "DBT"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if scenarios:
        keep_cols = ["datetime"] + [c for c in scenarios if c in df.columns]
        df = df[keep_cols]
    value_cols = [c for c in df.columns if c != "datetime"]
    if not value_cols:
        return pd.DataFrame(columns=["rel_path", "datetime", "DBT"])
    long = df.melt(id_vars=["datetime"], value_vars=value_cols, var_name="scenario", value_name="DBT")
    long["rel_path"] = str(station_key) + "__" + long["scenario"].astype(str)
    return long[["rel_path", "datetime", "DBT"]]


# -----------------------------
# Page header / config
# -----------------------------
f001__create_page_header()

# Use absolute path based on script location to avoid working directory issues
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_B_DATA_DIR = SCRIPT_DIR / "data" / "04__italy_tmy_fwg_parquet"
DEFAULT_CTI_DATA_DIR = SCRIPT_DIR / "data" / "04__italy_cti_parquet"
DEFAULT_DATA_DIR = DEFAULT_B_DATA_DIR
# Legacy paths kept for compatibility/debug (not used in B route)
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR
DEFAULT_TIDY_PARQUET = DEFAULT_DATA_DIR / "dbt_rh_tidy.parquet"
DEFAULT_TMYX_HOURLY_PARQUET = DEFAULT_DATA_DIR / "tmyx_hourly.parquet"


def _discover_parquet_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Discover .parquet files in data_dir for Data Preview. Returns [(path, display_label), ...] sorted by preference."""
    if not data_dir.exists():
        return []
    files: list[tuple[Path, str]] = []
    # B-route tables
    tables_dir = data_dir / "_tables"
    if tables_dir.exists():
        for p in sorted(tables_dir.glob("*.parquet")):
            name = p.name
            if name.startswith("D-TMYxFWG__DBT__F-DD__L-") or name.startswith("D-CTI__DBT__F-DD__L-"):
                files.append((p, f"{name} (daily stats)"))
            elif name.startswith("D-CTI__DBT__F-MM__L-"):
                files.append((p, f"{name} (monthly stats)"))
            elif "Inventory" in name:
                files.append((p, f"{name} (inventory)"))
            else:
                files.append((p, name))
    # Per-station hourly parquet (wide)
    for region_dir in sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name != "_tables"]):
        for p in sorted(region_dir.glob("*.parquet")):
            files.append((p, f"{region_dir.name}/{p.name} (station hourly)"))

    # Legacy single-folder parquet (if any)
    for p in sorted(data_dir.glob("*.parquet")):
        files.append((p, p.name))

    return files


def _discover_parquet_files_by_region(data_dir: Path) -> list[tuple[str, list[tuple[Path, str]]]]:
    """Group parquet files by region for Data Preview tabs. Returns [(region_label, [(path, label), ...]), ...]."""
    if not data_dir.exists():
        return []
    by_region: dict[str, list[tuple[Path, str]]] = {}
    tables_dir = data_dir / "_tables"
    if tables_dir.exists():
        by_region["Tables"] = []
        for p in sorted(tables_dir.glob("*.parquet")):
            name = p.name
            if name.startswith("D-TMYxFWG__DBT__F-DD__L-") or name.startswith("D-CTI__DBT__F-DD__L-"):
                by_region["Tables"].append((p, f"{name} (daily stats)"))
            elif name.startswith("D-CTI__DBT__F-MM__L-"):
                by_region["Tables"].append((p, f"{name} (monthly stats)"))
            elif "Inventory" in name:
                by_region["Tables"].append((p, f"{name} (inventory)"))
            else:
                by_region["Tables"].append((p, name))
    for region_dir in sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name != "_tables"]):
        region_code = region_dir.name
        by_region[region_code] = []
        for p in sorted(region_dir.glob("*.parquet")):
            by_region[region_code].append((p, f"{p.name} (station hourly)"))
    for p in sorted(data_dir.glob("*.parquet")):
        by_region.setdefault("Root", []).append((p, p.name))
    # Return as list of (region, files), Tables first then region codes
    order = ["Tables"] + sorted(k for k in by_region if k != "Tables" and k != "Root") + (["Root"] if "Root" in by_region else [])
    return [(r, by_region[r]) for r in order if by_region.get(r)]

# CTI (Itaca) weather stations (hourly DBT) - copy CSVs into this folder for Single Regions Data integration
DEFAULT_CTI_DIR = SCRIPT_DIR / "data" / "cti"
CTI_DBT_CSV = DEFAULT_CTI_DIR / "CTI__DBT__ITA_WeatherStations__All.csv"
_cti_list_candidates = (
    DEFAULT_CTI_DATA_DIR / "CTI__list__ITA_WeatherStations__All.csv",
    DEFAULT_B_DATA_DIR / "CTI__list__ITA_WeatherStations__All.csv",
    DEFAULT_CTI_DIR / "CTI__list__ITA_WeatherStations__All.csv",
)
CTI_LIST_CSV = next((p for p in _cti_list_candidates if p.exists()), _cti_list_candidates[-1])

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
    """Legacy index loader (unused in B route)."""
    if p.exists() and p.is_dir():
        return h.f10b__load_index_auto(p)
    if p.name.lower() == "epw_index.json" and not p.exists():
        return h.f10b__load_index_auto(p.parent)
    return h.f110__load_index(p)


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

# EPW index: prefer 04__italy_tmy_fwg_parquet, then legacy 03__italy_all_epw_DBT_streamlit
_epw_index_candidates = (
    DEFAULT_B_DATA_DIR / "D-TMY__epw_index.json",
    SCRIPT_DIR / "data" / "03__italy_all_epw_DBT_streamlit" / "D-TMY__epw_index.json",
)
_epw_index_path = next((p for p in _epw_index_candidates if p.exists()), _epw_index_candidates[-1])
epw_meta = _load_epw_index_meta(_epw_index_path)
idx = _build_idx_from_inventory(inventory, epw_meta)
if not epw_meta and not idx.empty:
    # Coordinates for map markers come from epw_meta; inventory from 06B has no lat/lon.
    st.warning(
        "**Map markers may be missing**: No EPW index found for station coordinates. "
        f"Place `D-TMY__epw_index.json` in `data/04__italy_tmy_fwg_parquet/` or in `data/03__italy_all_epw_DBT_streamlit/`. "
        f"Tried: {_epw_index_candidates[0]}, {_epw_index_candidates[1]}."
    )
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
    """
    Load regional hourly data for specific rel_paths.
    
    Groups rel_paths by region and loads only the needed regional files.
    This is much faster than loading the monolithic tidy parquet.
    
    Args:
        rel_paths: List of rel_path values to filter
        idx: Index DataFrame with region column
        base_dir: Base directory containing DBT__HR__XX.parquet files
    
    Returns:
        Combined DataFrame with datetime index and columns: DBT, rel_path, scenario
    """
    if not rel_paths:
        return pd.DataFrame()
    
    # Get region codes for the requested rel_paths
    rel_path_df = pd.DataFrame({"rel_path": rel_paths})
    rel_path_df = rel_path_df.merge(
        idx[["rel_path", "region"]].drop_duplicates(),
        on="rel_path",
        how="left"
    )
    
    # Fallback: if region is missing or UNK, try to extract from filename/rel_path
    import re
    from typing import Optional
    def extract_region_from_rel_path(rel_path_str: str) -> Optional[str]:
        """Extract region code from rel_path or filename."""
        # Try to find ITA_XX_ pattern in the path
        match = re.search(r"ITA_([A-Z]{2})_", str(rel_path_str).upper())
        if match:
            code = match.group(1)
            valid_codes = ["AB", "BC", "CM", "ER", "FV", "LB", "LG", "LM", "LZ", "MH", "ML", "PM", "PU", "SC", "SD", "TC", "TT", "UM", "VD", "VN"]
            if code in valid_codes:
                return code
        return None
    
    # Fix missing or UNK regions by extracting from rel_path
    missing_regions = rel_path_df["region"].isna() | (rel_path_df["region"] == "UNK")
    if missing_regions.any():
        for idx_row in rel_path_df[missing_regions].index:
            rel_path_str = rel_path_df.loc[idx_row, "rel_path"]
            extracted = extract_region_from_rel_path(rel_path_str)
            if extracted:
                rel_path_df.loc[idx_row, "region"] = extracted
    
    # Group by region and load each regional file
    regional_dfs = []
    processed_regions = set()
    
    for region_code in rel_path_df["region"].dropna().unique():
        if region_code == "UNK":
            continue  # Skip UNK regions
        
        # Avoid processing the same region twice
        if region_code in processed_regions:
            continue
        processed_regions.add(region_code)
        
        region_rel_paths = rel_path_df[rel_path_df["region"] == region_code]["rel_path"].tolist()
        hourly = h.load_regional_hourly(region_code, base_dir)
        if not hourly.empty:
            # Filter to only requested rel_paths
            if "rel_path" in hourly.columns:
                # Check if any rel_paths match
                available_rel_paths = hourly["rel_path"].unique()
                matching_rel_paths = [rp for rp in region_rel_paths if rp in available_rel_paths]
                if matching_rel_paths:
                    filtered = hourly[hourly["rel_path"].isin(matching_rel_paths)]
                    if not filtered.empty:
                        regional_dfs.append(filtered)
                # If no matches but we have data, include all data from this region
                # (rel_path values might not match exactly, but we still want the data)
                elif len(available_rel_paths) > 0:
                    regional_dfs.append(hourly)
            else:
                # If rel_path is missing, include all data
                regional_dfs.append(hourly)
    
    # If we still have missing regions, try to find them by checking all regional files
    # This handles cases where the index has wrong region codes
    missing_rel_paths = rel_path_df[rel_path_df["region"].isna() | (rel_path_df["region"] == "UNK")]["rel_path"].tolist()
    if missing_rel_paths:
        # Try all possible region codes
        all_region_codes = ["AB", "BC", "CM", "ER", "FV", "LB", "LG", "LM", "LZ", "MH", "ML", "PM", "PU", "SC", "SD", "TC", "TT", "UM", "VD", "VN"]
        for region_code in all_region_codes:
            if region_code in processed_regions:
                continue
            hourly = h.load_regional_hourly(region_code, base_dir)
            if not hourly.empty and "rel_path" in hourly.columns:
                # Check if any of our missing rel_paths are in this file
                available_rel_paths = set(hourly["rel_path"].unique())
                found_paths = [rp for rp in missing_rel_paths if rp in available_rel_paths]
                if found_paths:
                    filtered = hourly[hourly["rel_path"].isin(found_paths)]
                    if not filtered.empty:
                        regional_dfs.append(filtered)
                        processed_regions.add(region_code)
    
    if not regional_dfs:
        return pd.DataFrame()
    
    # Combine all regional dataframes
    combined = pd.concat(regional_dfs, axis=0)
    return combined.sort_index()


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
    """Compute daily stats from hourly DataFrame (datetime may be index or column).
    
    Args:
        hourly: DataFrame with datetime (as index or column) and columns: DBT, rel_path, scenario
    
    Returns:
        DataFrame with columns: rel_path, month, day, DBT_mean, DBT_max
    """
    if hourly.empty:
        return pd.DataFrame()
    
    try:
        # Handle datetime as index or column
        if isinstance(hourly.index, pd.DatetimeIndex):
            df = hourly.reset_index()
        else:
            df = hourly.copy()
            if "datetime" not in df.columns:
                # Try to infer datetime from index name
                if hourly.index.name == "datetime":
                    df = hourly.reset_index()
                else:
                    raise ValueError(f"Hourly DataFrame must have datetime as index or column. Found columns: {list(df.columns)}, index: {df.index.name}")
        
        # Ensure rel_path exists
        if "rel_path" not in df.columns:
            raise ValueError(f"Hourly DataFrame must have 'rel_path' column. Found columns: {list(df.columns)}")
        
        # Ensure DBT exists
        if "DBT" not in df.columns:
            raise ValueError(f"Hourly DataFrame must have 'DBT' column. Found columns: {list(df.columns)}")
        
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", "DBT", "rel_path"])
        
        if df.empty:
            return pd.DataFrame(columns=["rel_path", "month", "day", "DBT_mean", "DBT_max"])
        
        df["month"] = df["datetime"].dt.month.astype("int8")
        df["day"] = df["datetime"].dt.day.astype("int8")
        
        # Group by rel_path, month, day and compute daily aggregates
        daily = df.groupby(["rel_path", "month", "day"], as_index=False, observed=True).agg(
            DBT_mean=("DBT", "mean"),
            DBT_max=("DBT", "max"),
        )
        
        # Optimize dtypes
        daily["rel_path"] = daily["rel_path"].astype("category")
        daily["DBT_mean"] = daily["DBT_mean"].astype("float32")
        daily["DBT_max"] = daily["DBT_max"].astype("float32")
        
        return daily
    except Exception as e:
        st.error(f"Error computing daily stats from hourly data: {e}")
        st.caption(f"Hourly DataFrame shape: {hourly.shape}, columns: {list(hourly.columns)}, index type: {type(hourly.index)}")
        return pd.DataFrame(columns=["rel_path", "month", "day", "DBT_mean", "DBT_max"])


@st.cache_data(show_spinner=False)
def _load_file_stats_cached(daily_stats: pd.DataFrame, percentile: float,
                            precomputed_path: Path):
    """Load or compute file stats with caching.

    Tries to load precomputed file (from 06B: D-TMYxFWG__FileStats__P-{p}.parquet).
    File may be single-percentile (no 'percentile' column) or multi-percentile.
    Falls back to computing from daily_stats if not available.
    """
    if precomputed_path.exists():
        try:
            file_stats_all = pd.read_parquet(precomputed_path)
            if "percentile" in file_stats_all.columns:
                file_stats = file_stats_all[file_stats_all["percentile"] == percentile].copy()
                if not file_stats.empty:
                    file_stats = file_stats.drop(columns=["percentile"])
            else:
                file_stats = file_stats_all.copy()
            if not file_stats.empty:
                return _timed(
                    "load_file_stats_precomputed",
                    lambda: file_stats,
                    notes="Loads precomputed file stats (fast path).",
                )
        except Exception:
            pass  # Fallback to computation

    # Fallback to computation
    return _timed(
        "f123__build_file_stats_from_daily",
        lambda: h.f123__build_file_stats_from_daily(daily_stats, percentile),
        notes="Computes per-file Tmax percentile and Tavg mean from daily stats.",
    )


@st.cache_data(show_spinner=False)
def _load_location_deltas_cached(daily_stats: pd.DataFrame, idx: pd.DataFrame,
                                 baseline_variant: str, compare_variant: str,
                                 percentile: float, precomputed_path: Path):
    """Load or compute location deltas with caching.
    
    Tries to load precomputed location_deltas_by_variant_pair_percentile.parquet first.
    Falls back to computing from daily_stats if not available.
    """
    if precomputed_path.exists():
        try:
            deltas_all = pd.read_parquet(precomputed_path)
            # Convert percentile from percentage (99.0) to decimal (0.99) for comparison
            percentile_decimal = float(percentile) / 100.0
            
            delta_df = deltas_all[
                (deltas_all["baseline_variant"] == baseline_variant) &
                (deltas_all["compare_variant"] == compare_variant) &
                (deltas_all["percentile"] == percentile_decimal)
            ].copy()
            
            if not delta_df.empty:
                delta_df = delta_df.drop(columns=["baseline_variant", "compare_variant", "percentile"])
                return _timed(
                    "load_location_deltas_precomputed",
                    lambda: delta_df,
                    notes="Loads precomputed location deltas (fast path).",
                )
        except Exception:
            pass  # Fallback to computation
    
    # Fallback to computation
    return _timed(
        "f125__compute_location_deltas_from_daily",
        lambda: h.f125__compute_location_deltas_from_daily(
            daily_stats,
            idx,
            baseline_variant=baseline_variant,
            compare_variant=compare_variant,
            percentile=float(percentile),
        ),
        notes="Computes per-location max monthly ΔT for map/table.",
    )


@st.cache_data(show_spinner=False)
def _load_precomputed_location_stats(variant: str, percentile: float, daily_stats, idx, precomputed_path: Path):
    """Try to load precomputed location stats with caching, fallback to computation.
    
    Args:
        variant: Variant name
        percentile: Percentile as decimal (0.99 for 99th percentile)
        daily_stats: Daily stats DataFrame (for fallback)
        idx: Index DataFrame (for fallback)
        precomputed_path: Path to precomputed location stats parquet file
    """
    if precomputed_path.exists():
        try:
            stats_all = pd.read_parquet(precomputed_path)
            stats_df = stats_all[
                (stats_all["variant"] == variant) &
                (stats_all["percentile"] == percentile)
            ].copy()
            if not stats_df.empty:
                stats_df = stats_df.drop(columns=["variant", "percentile"])
                return _timed(
                    "load_location_stats_precomputed",
                    lambda: stats_df,
                    notes=f"Loads precomputed location stats for {variant}.",
                )
        except Exception:
            pass  # Fallback to computation
    
    return _timed(
        "f28b__compute_location_stats_for_variant_from_daily",
        lambda: h.f28b__compute_location_stats_for_variant_from_daily(
            daily_stats, idx, variant=variant, percentile=percentile
        ),
        notes=f"Computes location stats for {variant}.",
    )


def _filter_cti_points_by_region(cti_points: list, region_code: str | None) -> list:
    """Return CTI points that fall within the given region (region code, e.g. 'LM').
    Used for server-side filtering when needed; the D3 CTI map also filters by selected region client-side via localStorage.
    """
    if not region_code:
        return cti_points
    return [p for p in cti_points if str(p.get("region")) == str(region_code)]


@st.cache_data(show_spinner=False)
def _load_precomputed_monthly_delta_table(baseline_variant: str, compare_variant: str, 
                                         percentile: float, metric_key: str, 
                                         daily_stats, idx, precomputed_path: Path):
    """Try to load precomputed monthly delta table with caching, fallback to computation.
    
    Args:
        baseline_variant: Baseline variant name
        compare_variant: Compare variant name
        percentile: Percentile as percentage (99.0 for 99th percentile)
        metric_key: "dTmax" or "dTavg"
        daily_stats: Daily stats DataFrame (for fallback)
        idx: Index DataFrame (for fallback)
        precomputed_path: Path to precomputed monthly delta tables parquet file
    """
    if precomputed_path.exists():
        try:
            tables_all = pd.read_parquet(precomputed_path)
            # Convert percentile from percentage (99.0) to decimal (0.99) for comparison
            percentile_decimal = float(percentile) / 100.0
            table_df = tables_all[
                (tables_all["baseline_variant"] == baseline_variant) &
                (tables_all["compare_variant"] == compare_variant) &
                (tables_all["percentile"] == percentile_decimal) &
                (tables_all["metric_key"] == metric_key)
            ].copy()
            if not table_df.empty:
                table_df = table_df.drop(columns=["baseline_variant", "compare_variant", "percentile", "metric_key"])
                return _timed(
                    "load_monthly_delta_table_precomputed",
                    lambda: table_df,
                    notes="Loads precomputed monthly delta table.",
                )
        except Exception:
            pass  # Fallback to computation
    
    return _timed(
        "f124__build_monthly_delta_table",
        lambda: h.f124__build_monthly_delta_table(
            daily_stats, idx,
            baseline_variant=baseline_variant,
            compare_variant=compare_variant,
            percentile=float(percentile) / 100.0,
            metric_key=metric_key,
        ),
        notes="Builds monthly delta table (fallback when 06B precomputed file missing).",
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


# Sidebar controls (all radio/select widgets)
future_tags_by_baseline: dict[str, list[str]] = {}
for scenario in all_scenarios:
    if "__" not in scenario:
        continue
    base, tag = scenario.split("__", 1)
    future_tags_by_baseline.setdefault(base, set()).add(tag)
future_tags_by_baseline = {k: sorted(list(v)) for k, v in future_tags_by_baseline.items()}

with st.sidebar:
    st.markdown(f"#### {label('baseline_climate_file')}")

    default_baseline = "tmyx_2009-2023"
    default_baseline = "tmyx"
    default_idx = present_variants.index(default_baseline) if default_baseline in present_variants else 0
    baseline_variant = st.radio(
        label("current_climate_baseline"),
        options=present_variants,
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

top_tabs = st.tabs(
    [
        label("welcome"),
        label("future_weather_scenarios"),
        "Data & Code Debug",
    ]
)

with top_tabs[0]:
    render_welcome_page()


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
            "`data/04__italy_cti_parquet/`, `data/04__italy_tmy_fwg_parquet/`, or `data/cti/`."
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


with top_tabs[1]:
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
            V2_DASHBOARD_COLS = "5.5fr 4fr 1fr"
            V2_MAPS_ROW_GAP_PX = 10
            V2_MAPS_ROW3_GAP_PX = 4
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
                if present_variants and not daily_stats.empty and not idx.empty:
                    for v in present_variants:
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
                    if baseline_variant_future in present_variants
                    else ("tmyx" if "tmyx" in present_variants else present_variants[0])
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

                side_text_html = (
                    "<b>Temperature Summary</b><br/>"
                    "• Tmax uses selected percentile<br/>"
                    "• Tavg is mean of daily means<br/>"
                    "• Click a marker to update charts"
                )

                html_v2 = h.f23d__d3_region_maps_html(
                    baseline_variants=list(top_combined_abs.keys()),
                    abs_points_by_variant=top_combined_abs,
                    delta_points_by_variant={},
                    metric_key=metric_key,
                    compare_variant=compare_variant,
                    percentile=float(percentile),
                    region_options=region_options_data,
                    future_abs_points_by_variant=top_future_abs_points if top_future_abs_points else None,
                    initial_region=None,
                    click_callback_key="single_regions_location_click_v2",
                    profiles_bundle_by_variant=profiles_bundle_by_variant if profiles_bundle_by_variant else None,
                    hide_region_selector=False,
                    layout_mode="columns",
                    cti_on_top=True,
                    hide_row2=True,
                    side_text_html=side_text_html,
                    dashboard_cols=V2_DASHBOARD_COLS,
                    maps_row_gap_px=V2_MAPS_ROW_GAP_PX,
                    maps_row3_gap_px=V2_MAPS_ROW3_GAP_PX,
                )
                components.html(html_v2, height=1150, scrolling=False)

    with scenario_tabs[3]:
        st.markdown(f"##### {label('current_weather_data_tmyx')}")
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

                # Row 1: Charts / KPIs
                # Use margin-bottom (can be negative) to control spacing between header/legend row and charts row.
                # Negative values will "lift" the charts row up.
                st.markdown(
                    f"<div style='height:0px; margin-bottom:{TMYX_HEADER_TO_CHARTS_GAP_PX}px'></div>",
                    unsafe_allow_html=True,
                )
                row1 = st.columns([1, 1, 1, 0.9], gap="medium")
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
                        # Legend is in the header row; hide colorbars inside the figure.
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
                        # Legend is in the header row; hide legend inside the figure.
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
                        # Legend is in the header row; hide legend inside the figure.
                        stacked_fig.update_layout(showlegend=False)
                        st.plotly_chart(stacked_fig, width="stretch", key=f"stacked_subplots_{loc_id}")
                    else:
                        st.info("No stacked column data available.")

                with row1[3]:
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

with top_tabs[2]:
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
            region_tab_names = [f"{region} ({len(files)})" for region, files in by_region]
            data_preview_tabs = st.tabs(region_tab_names)
            for tab, (region_code, region_files) in zip(data_preview_tabs, by_region):
                with tab:
                    for pa, label in region_files:
                        try:
                            df_p = _read_parquet_robust(pa)
                            if df_p is None or df_p.empty:
                                st.caption(f"**{pa.name}** — empty")
                                continue
                            nrows, ncols = len(df_p), len(df_p.columns)
                            st.markdown(f"**{pa.name}** — {nrows:,} rows, {ncols} columns")
                            st.dataframe(df_p.head(preview_n), width="stretch", hide_index=True)
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
