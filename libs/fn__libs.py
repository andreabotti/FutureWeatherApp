
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import calendar
import altair as alt
import numpy as np
import pandas as pd

def _ensure_lat_lon_columns(idx: pd.DataFrame) -> pd.DataFrame:
    """Ensure idx has both latitude/longitude and lat/lon columns.

    Your new index writers may use (lat, lon) while app code expects (latitude, longitude).
    This normalizes columns so downstream code never KeyErrors.
    """
    if idx is None or idx.empty:
        return idx

    if "latitude" not in idx.columns and "lat" in idx.columns:
        idx["latitude"] = pd.to_numeric(idx["lat"], errors="coerce")
    elif "latitude" in idx.columns:
        idx["latitude"] = pd.to_numeric(idx["latitude"], errors="coerce")
    else:
        idx["latitude"] = pd.NA

    if "longitude" not in idx.columns and "lon" in idx.columns:
        idx["longitude"] = pd.to_numeric(idx["lon"], errors="coerce")
    elif "longitude" in idx.columns:
        idx["longitude"] = pd.to_numeric(idx["longitude"], errors="coerce")
    else:
        idx["longitude"] = pd.NA

    # Keep aliases both ways
    if "lat" not in idx.columns and "latitude" in idx.columns:
        idx["lat"] = idx["latitude"]
    if "lon" not in idx.columns and "longitude" in idx.columns:
        idx["lon"] = idx["longitude"]

    return idx


def _bool_is_cti(idx: pd.DataFrame) -> pd.Series:
    """Robust CTI flag used throughout pairing and exclusions."""
    if idx is None or idx.empty:
        return pd.Series([], dtype=bool)

    if "is_cti" in idx.columns:
        return idx["is_cti"].fillna(False).astype(bool)

    scen = idx["scenario"].astype(str) if "scenario" in idx.columns else pd.Series([""] * len(idx))
    src = idx["source"].astype(str) if "source" in idx.columns else pd.Series([""] * len(idx))
    ds = idx["dataset"].astype(str) if "dataset" in idx.columns else pd.Series([""] * len(idx))

    return (
        ds.str.upper().eq("CTI")
        | scen.str.contains("cti", case=False, na=False)
        | src.str.contains("cti", case=False, na=False)
    )
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from ladybug.color import Colorset
    HAS_LADYBUG = True
except ImportError:
    HAS_LADYBUG = False


def _rgb_to_hex(rgb_tuple):
    """Convert RGB tuple to hex color string."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))


def _get_nuanced_colorset():
    """Get the nuanced colorset as a list of hex colors."""
    if HAS_LADYBUG:
        colorset = Colorset.nuanced()
        return [_rgb_to_hex((c.r, c.g, c.b)) for c in colorset]
    else:
        # Fallback: use a similar color palette
        import plotly.express as px
        return px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])


def _get_nuanced_colorscale():
    """Convert nuanced colorset to Plotly colorscale format."""
    colors = _get_nuanced_colorset()
    n = len(colors)
    if n == 0:
        return "RdBu_r"
    # Create colorscale: [[0, color1], [0.5, color2], ..., [1, colorN]]
    return [[i / (n - 1) if n > 1 else 0, color] for i, color in enumerate(colors)]


def f101__inject_inter_font() -> None:
    """
    Apply Inter as the default font across Streamlit UI.
    (Charts may require separate font settings; we handle those separately too.)
    """
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0');

html, body, .stApp {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

/* Serif headings to match title styling */
h1, h2, h3, h4,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
  font-family: 'Lora', 'Source Serif Pro', 'Source Serif 4', serif;
}

/* Apply Inter broadly but DO NOT clobber Streamlit's icon fonts (Material Symbols). */
.stApp *:not(.material-icons):not(.material-symbols-outlined):not(.material-symbols-rounded):not(.material-symbols-sharp):not([data-testid="stIconMaterial"]) {
  font-family: inherit;
}

/* Force icon fonts for Streamlit Material icon spans (fixes 'keyboard_arrow_*' showing as text). */
.material-symbols-outlined,
span[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Outlined" !important;
  font-weight: normal !important;
  font-style: normal !important;
  line-height: 1 !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  white-space: nowrap !important;
  direction: ltr !important;
  -webkit-font-feature-settings: "liga" !important;
  -webkit-font-smoothing: antialiased !important;
}

code, pre, kbd, samp {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
}

/* Reduce sidebar font size by 20% */
[data-testid="stSidebar"] {
  font-size: 0.8rem !important;
}

/* Replace red primary color with EETRA Light Green (#434f3d) for buttons and widgets */
/* Note: EETRA Green (#33b24a) is kept for chart traces */
.stButton > button {
  background-color: #434f3d !important;
  color: white !important;
  border: none !important;
}

.stButton > button:hover {
  background-color: #363e32 !important;
  border: none !important;
}

.stButton > button:focus {
  background-color: #363e32 !important;
  border: none !important;
  box-shadow: 0 0 0 0.2rem rgba(67, 79, 61, 0.25) !important;
}

.stDownloadButton > button {
  background-color: #434f3d !important;
  color: white !important;
  border: none !important;
}

.stDownloadButton > button:hover {
  background-color: #363e32 !important;
  border: none !important;
}

.stDownloadButton > button:focus {
  background-color: #363e32 !important;
  border: none !important;
  box-shadow: 0 0 0 0.2rem rgba(67, 79, 61, 0.25) !important;
}

/* Radio buttons and other widgets */
.stRadio > div > label {
  color: #434f3d !important;
}

.stRadio > div > label[data-baseweb="radio"] > div:first-child {
  background-color: #434f3d !important;
}

/* Selectbox and other select widgets */
.stSelectbox > div > div {
  border-color: #434f3d !important;
}

.stSelectbox > div > div:focus-within {
  border-color: #434f3d !important;
  box-shadow: 0 0 0 0.2rem rgba(67, 79, 61, 0.25) !important;
}

/* Progress bars and other primary-colored elements */
.stProgress > div > div > div {
  background-color: #434f3d !important;
}

/* Links */
a {
  color: #434f3d !important;
}

a:hover {
  color: #363e32 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def f102__enable_altair_inter_theme() -> None:
    """
    Make Altair use Inter so charts match the app typography.
    """

    def _theme():
        return {
            "config": {
                "title": {"font": "Inter", "fontSize": 14},
                "axis": {
                    "labelFont": "Inter",
                    "titleFont": "Inter",
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
                "legend": {
                    "labelFont": "Inter",
                    "titleFont": "Inter",
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
            }
        }

    try:
        alt.themes.register("inter", _theme)
    except Exception:
        pass
    alt.themes.enable("inter")


# -----------------------------
# Variant / id helpers
# -----------------------------
def f103__parse_variant_from_filename(filename: str) -> str:
    """
    Extract a compact variant label from EPW filename.

    Present/baseline examples:
      ..._TMYx.epw                -> tmyx
      ..._TMYx.2007-2021.epw      -> tmyx_2007-2021

    Morphed examples:
      ..._Ensemble_rcp26_2050.epw -> rcp26_2050
    """
    fn = (filename or "").lower()

    tmy = re.search(r"(tmyx)(?:\.(\d{4}-\d{4}))?", fn)
    if tmy:
        period = tmy.group(2)
        return f"tmyx_{period}" if period else "tmyx"

    rcp = re.search(r"(rcp\d{2})", fn)
    year = re.search(r"(20\d{2})", fn)
    if rcp and year:
        return f"{rcp.group(1)}_{year.group(1)}"

    return "unknown"


def f104__parse_station_id_from_filename(filename: str) -> str | None:
    """
    Extract station id from filenames like:
      ITA_AB_Fucino.162270_TMYx.epw -> 162270
    """
    m = re.search(r"\.(\d{5,})_", filename or "")
    return m.group(1) if m else None


def f04b__parse_station_id_from_group_key(group_key: str) -> str | None:
    """
    Extract station id from group_key (folder name / baseline stem) for FWG morphed files.
    e.g. ITA_AB_Fucino.162270_TMYx -> 162270
    """
    if not group_key:
        return None
    m = re.search(r"\.(\d{5,})_", group_key)
    if m:
        return m.group(1)
    m = re.search(r"(\d{5,})", group_key)
    return m.group(1) if m else None


def f105__is_present_variant(v: str) -> bool:
    return isinstance(v, str) and v.startswith("tmyx")


def f106__present_variant_sort_key(v: str):
    if v == "tmyx":
        return (0, 0, 0)
    m = re.match(r"^tmyx_(\d{4})-(\d{4})$", v or "")
    if m:
        return (1, int(m.group(1)), int(m.group(2)))
    return (2, 9999, 9999)


def f107__group_key_from_filename(filename: str) -> str:
    """
    Stable key that matches FWG output folder names.
    """
    return Path(filename).stem


def f07b__group_key_from_rel_path(rel_path: str, filename: str) -> str:
    """
    Prefer FWG folder group when rel_path is from 02__italy_fwg_outputs:
      .../02__italy_fwg_outputs/<REGION>/<GROUP>/<file>.epw
    Fallback to filename stem.
    """
    rel_norm = str(rel_path or "").replace("\\", "/")
    parts = [p for p in rel_norm.split("/") if p]
    try:
        i = parts.index("02__italy_fwg_outputs")
        if i + 2 < len(parts):
            return parts[i + 2]
    except ValueError:
        pass
    return f107__group_key_from_filename(filename)


def f108__normalize_group_key(group: Any, filename: str) -> str:
    """
    Use index record group if present, otherwise fall back to filename stem.
    Also fixes legacy group values that were stored as "<file>.epw".
    """
    if isinstance(group, str) and group:
        if group.lower().endswith(".epw"):
            return Path(group).stem
        return group
    return f107__group_key_from_filename(filename)


# -----------------------------
# Data loading / processing
# -----------------------------
@st.cache_data(show_spinner=False)
def f109__load_index_records(index_path: Path) -> list[dict]:
    with open(index_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("epw_index.json must contain a JSON list of records.")
    return records


@st.cache_data(show_spinner=False)
def f09b__discover_index_paths(base_dir: Path) -> list[Path]:
    """
    New naming:
      - D-TMY__epw_index.json
      - D-RCP__epw_index.json
      - D-CTI__epw_index.json
    Legacy:
      - epw_index.json
    """
    base_dir = Path(base_dir)
    legacy = base_dir / "epw_index.json"
    if legacy.exists():
        return [legacy]
    paths = sorted(base_dir.glob("D-*__epw_index.json"))
    return paths


@st.cache_data(show_spinner=False)
def f10b__load_index_auto(base_dir: Path) -> pd.DataFrame:
    """
    Load index from either legacy epw_index.json or new per-dataset D-*__epw_index.json,
    concatenating if multiple are present.
    """
    paths = f09b__discover_index_paths(base_dir)
    if not paths:
        raise FileNotFoundError(
            f"No index JSON found in {base_dir}. Expected epw_index.json or D-*__epw_index.json"
        )

    records: list[dict] = []
    for p in paths:
        records.extend(f109__load_index_records(p))

    idx = pd.DataFrame(records)

    if "dataset" not in idx.columns:
        idx["dataset"] = "UNK"

    idx["variant"] = idx["filename"].apply(f103__parse_variant_from_filename)

    group_vals = []
    for g, fn, rp in zip(
        idx.get("group", [None] * len(idx)),
        idx.get("filename", [""] * len(idx)),
        idx.get("rel_path", [""] * len(idx)),
    ):
        if isinstance(g, str) and g:
            group_vals.append(f108__normalize_group_key(g, fn))
        else:
            group_vals.append(f07b__group_key_from_rel_path(rp, fn))
    idx["group_key"] = group_vals

    wmo_s = idx.get("wmo", pd.Series([pd.NA] * len(idx))).astype(str).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    station_id_fn = idx["filename"].apply(f104__parse_station_id_from_filename)
    group_station_id = idx["group_key"].apply(f04b__parse_station_id_from_group_key)
    filename_stem = idx["filename"].apply(lambda fn: Path(fn).stem if isinstance(fn, str) else pd.NA)
    scenario_s = idx.get("scenario", pd.Series([pd.NA] * len(idx))).astype(str)
    idx["location_id"] = (
        wmo_s.fillna(station_id_fn).fillna(group_station_id).fillna(filename_stem).fillna(scenario_s).astype(str)
    )

    dataset_s = idx.get("dataset", pd.Series([""] * len(idx))).astype(str).str.upper()
    source_s = idx.get("source", pd.Series([""] * len(idx))).astype(str)
    idx["is_cti"] = (
        dataset_s.eq("CTI")
        | source_s.str.contains("cti", case=False, na=False)
        | scenario_s.str.contains("cti", case=False, na=False)
    )
    return idx


@st.cache_data(show_spinner=False)
def f110__load_index(index_path: Path) -> pd.DataFrame:
    records = f109__load_index_records(index_path)
    idx = pd.DataFrame(records)

    idx["variant"] = idx["filename"].apply(f103__parse_variant_from_filename)

    # group_key first (folder name / baseline stem) — used for pairing and location_id fallback
    group_vals = []
    for g, fn, rp in zip(
        idx.get("group", [None] * len(idx)),
        idx.get("filename", [""] * len(idx)),
        idx.get("rel_path", [""] * len(idx)),
    ):
        if isinstance(g, str) and g:
            group_vals.append(f108__normalize_group_key(g, fn))
        else:
            group_vals.append(f07b__group_key_from_rel_path(rp, fn))
    idx["group_key"] = group_vals

    # location_id priority: wmo → station_id from filename → group_station_id → filename stem → scenario
    wmo_s = idx.get("wmo", pd.Series([pd.NA] * len(idx))).astype(str).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    station_id_fn = idx["filename"].apply(f104__parse_station_id_from_filename)
    group_station_id = idx["group_key"].apply(f04b__parse_station_id_from_group_key)
    filename_stem = idx["filename"].apply(lambda fn: Path(fn).stem if isinstance(fn, str) else pd.NA)
    scenario_s = idx.get("scenario", pd.Series([pd.NA] * len(idx))).astype(str)
    idx["location_id"] = (
        wmo_s.fillna(station_id_fn).fillna(group_station_id).fillna(filename_stem).fillna(scenario_s).astype(str)
    )

    # CTI exclusion: source or scenario contains "cti" (do not rely on exact "cti" tag)
    source_s = idx.get("source", pd.Series([""] * len(idx))).astype(str)
    idx["is_cti"] = (
        source_s.str.contains("cti", case=False, na=False)
        | scenario_s.str.contains("cti", case=False, na=False)
    )
    return idx


@st.cache_data(show_spinner=False)
def load_regional_hourly(region_code: str, base_dir: Path, dataset: str | None = None) -> pd.DataFrame:
    """
    Load regional hourly DBT data.

    NEW naming:
      D-<DATASET>__DBT__F-HR__L-<REGION>.parquet
      e.g. D-RCP__DBT__F-HR__L-AB.parquet

    Legacy: DBT__HR__<REGION>.parquet

    If dataset is None, loads ALL datasets for that region and concatenates them.
    Returns empty DataFrame if nothing exists.
    """
    base_dir = Path(base_dir)
    region_code = str(region_code).upper()

    if dataset:
        ds = str(dataset).upper()
        candidates = [
            base_dir / f"D-{ds}__DBT__F-HR__L-{region_code}.parquet",
            base_dir / f"DBT__HR__{region_code}.parquet",
        ]
    else:
        candidates = sorted(base_dir.glob(f"D-*__DBT__F-HR__L-{region_code}.parquet"))
        if not candidates:
            legacy = base_dir / f"DBT__HR__{region_code}.parquet"
            if legacy.exists():
                candidates = [legacy]

    if not candidates:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                continue

            if isinstance(df.index, pd.DatetimeIndex):
                df.index.name = "datetime"
            elif "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                try:
                    df.index = pd.to_datetime(df.index, errors="coerce")
                    df = df[~df.index.isna()]
                    df.index.name = "datetime"
                except Exception:
                    continue

            if "dataset" not in df.columns:
                m = re.match(r"^D-([^_]+)__DBT__F-HR__L-", path.name, flags=re.IGNORECASE)
                df["dataset"] = m.group(1).upper() if m else "UNK"

            dfs.append(df)

        except Exception as e:
            st.error(f"Error loading {path.name}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, axis=0).sort_index()
    return out


def discover_regions(base_dir: Path) -> list[str]:
    """Discover region subfolders under the new B-route base directory."""
    if not base_dir.exists():
        return []
    regions = [
        p.name for p in base_dir.iterdir()
        if p.is_dir() and p.name != "_tables"
    ]
    return sorted(regions)


def discover_stations(base_dir: Path, region: str) -> list[str]:
    """List station keys for a region based on parquet filenames."""
    region_dir = base_dir / str(region)
    if not region_dir.exists():
        return []
    return sorted([p.stem for p in region_dir.glob("*.parquet")])


def load_station_hourly(base_dir: Path, region: str, station_key: str) -> pd.DataFrame:
    """Load a single station's hourly parquet (wide format)."""
    station_path = base_dir / str(region) / f"{station_key}.parquet"
    if not station_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(station_path)
    except Exception:
        return pd.DataFrame()


def load_cti_station_hourly(base_dir: Path, region: str, station_key: str) -> pd.DataFrame:
    """Load a single CTI station's hourly parquet (wide format)."""
    station_path = base_dir / str(region) / f"{station_key}.parquet"
    if not station_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(station_path)
    except Exception:
        return pd.DataFrame()


def load_daily_stats(base_dir: Path) -> pd.DataFrame:
    """Load precomputed daily stats table for TMYx↔FWG comparisons."""
    daily_path = base_dir / "_tables" / "D-TMYxFWG__DBT__F-DD__L-ALL.parquet"
    if not daily_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(daily_path)


def load_cti_daily_stats(base_dir: Path) -> pd.DataFrame:
    """Load precomputed daily stats table for CTI dataset."""
    daily_path = base_dir / "_tables" / "D-CTI__DBT__F-DD__L-ALL.parquet"
    if not daily_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(daily_path)


def load_cti_monthly_stats(base_dir: Path) -> pd.DataFrame:
    """Load precomputed monthly stats table for CTI dataset."""
    monthly_path = base_dir / "_tables" / "D-CTI__DBT__F-MM__L-ALL.parquet"
    if not monthly_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(monthly_path)


def load_inventory(base_dir: Path) -> pd.DataFrame:
    """Load station inventory table for available scenarios."""
    inv_path = base_dir / "_tables" / "D-TMYxFWG__Inventory__F-NA__L-ALL.parquet"
    if not inv_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(inv_path)


def load_cti_inventory(base_dir: Path) -> pd.DataFrame:
    """Load station inventory table for CTI dataset."""
    inv_path = base_dir / "_tables" / "D-CTI__Inventory__F-NA__L-ALL.parquet"
    if not inv_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(inv_path)


def load_pairing_debug(base_dir: Path) -> pd.DataFrame:
    """Load pairing debug CSV (missing/present scenario columns per station)."""
    debug_path = base_dir / "_tables" / "pairing_debug.csv"
    if not debug_path.exists():
        return pd.DataFrame()
    return pd.read_csv(debug_path)


def f111__load_tidy_parquet_raw(tidy_path: Path) -> pd.DataFrame:
    """Load raw tidy parquet file. Returns empty DataFrame if file doesn't exist."""
    if not tidy_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {tidy_path}")
    
    try:
        df = pd.read_parquet(tidy_path)
    except FileNotFoundError:
        # Re-raise FileNotFoundError as-is
        raise
    except Exception as e:
        # For other errors (like missing pyarrow), provide helpful message
        raise RuntimeError(
            "Could not read parquet. Install a parquet engine.\n\n"
            "Try:\n"
            "  pip install pyarrow\n\n"
            f"Original error: {repr(e)}"
        )
    return df


def f112__postprocess_tidy_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"])
    return out


@st.cache_data(show_spinner=False)
def f113__build_file_stats(tidy: pd.DataFrame) -> pd.DataFrame:
    return tidy.groupby("rel_path", as_index=False)["DBT"].agg(Tmax="max", Tavg="mean")


@st.cache_data(show_spinner=False)
def f123__build_file_stats_from_daily(daily_stats: pd.DataFrame, percentile: float) -> pd.DataFrame:
    """
    Build per-file stats from daily aggregates.

    - Tmax: percentile of daily max DBT (qP)
    - Tavg: mean of daily mean DBT

    CTI rows are excluded. For precomputed outputs use f123h__build_file_stats_from_hourly
    (percentiles from hourly = more robust).
    """
    if daily_stats.empty:
        return pd.DataFrame()
    
    # Column compatibility: handle both new (DBT_mean, DBT_max) and legacy names
    df = daily_stats.copy()
    
    # Check and normalize column names
    if "DBT_max" not in df.columns and "DBT_MAX" in df.columns:
        df = df.rename(columns={"DBT_MAX": "DBT_max"})
    if "DBT_mean" not in df.columns and "DBT_MEAN" in df.columns:
        df = df.rename(columns={"DBT_MEAN": "DBT_mean"})
    
    # Verify required columns exist
    if "DBT_max" not in df.columns or "DBT_mean" not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"daily_stats missing required columns. "
            f"Expected: ['DBT_max', 'DBT_mean'], Found: {available}. "
            f"Please re-run: python data/data_preparation_scripts/06_precompute_derived_stats.py"
        )
    
    p = max(0.0, min(1.0, float(percentile)))
    if "scenario" in df.columns:
        df = df[~df["scenario"].astype(str).str.contains("cti", case=False, na=False)].copy()
    g = df.groupby("rel_path", as_index=False, observed=True).agg(
        Tmax=("DBT_max", lambda s: s.quantile(p)),
        Tavg=("DBT_mean", "mean"),
    )
    return g


def f123h__build_file_stats_from_hourly(hourly: pd.DataFrame, percentile: float) -> pd.DataFrame:
    """
    Build per-file stats from hourly DBT (robust: percentiles from hourly values, not daily).

    - Tmax: percentile of all hourly DBT (qP) for each rel_path
    - Tavg: mean of all hourly DBT for each rel_path

    CTI rows are excluded (scenario contains 'cti') so file stats are TMYx/RCP only.
    Expects hourly with columns: DBT, rel_path; optional: scenario.
    """
    if hourly is None or hourly.empty:
        return pd.DataFrame()
    df = hourly.copy()
    if "DBT" not in df.columns or "rel_path" not in df.columns:
        return pd.DataFrame()
    if "scenario" in df.columns:
        df = df[~df["scenario"].astype(str).str.contains("cti", case=False, na=False)].copy()
    p = max(0.0, min(1.0, float(percentile)))
    g = df.groupby("rel_path", as_index=False, observed=True).agg(
        Tmax=("DBT", lambda s: s.quantile(p)),
        Tavg=("DBT", "mean"),
    )
    return g


def f124__build_monthly_delta_table(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    baseline_variant: str,
    compare_variant: str,
    percentile: float,
    metric_key: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a wide table with monthly deltas and max monthly delta for all locations.

    - ΔTmax: monthly percentile of daily max (qP) and delta (comp - base)
    - ΔTavg: monthly mean of daily mean and delta (comp - base)
    - "max yearly": max monthly delta across the 12 months
    
    NOTE: This function is for TMYx vs RCP comparisons only. CTI data should be excluded.
    """
    p = max(0.0, min(1.0, float(percentile)))
    use_max = metric_key == "dTmax"

    # Exclude CTI: source or scenario contains "cti"
    idx_no_cti = idx[~idx["is_cti"]].copy() if "is_cti" in idx.columns else idx.copy()
    if "scenario" in daily_stats.columns:
        sc = daily_stats["scenario"].astype(str)
        daily_stats = daily_stats[~sc.str.contains("cti", case=False, na=False)].copy()

    base_rows = idx_no_cti[idx_no_cti["variant"] == baseline_variant].copy()
    if base_rows.empty:
        return pd.DataFrame()
    base_rows["base_group"] = base_rows["group_key"]
    # Pair by group_key first so we match even when compare location_id differs (e.g. wrong index group)
    base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
        base_location_id=("location_id", "first"),
        base_group=("group_key", "first"),
    )

    comp_rows = idx_no_cti[idx_no_cti["variant"] == compare_variant].copy()
    if comp_rows.empty:
        return pd.DataFrame()
    comp_rows = comp_rows.merge(base_group_by_key, on="group_key", how="inner")
    used_fallback = False
    # Fallback: if no match by group_key (e.g. index has wrong group for FWG), try by location_id
    if comp_rows.empty:
        used_fallback = True
        base_group_by_loc = base_rows.groupby("location_id", as_index=False).agg(
            base_location_id=("location_id", "first"),
            base_group=("group_key", "first"),
        )
        comp_rows = idx_no_cti[idx_no_cti["variant"] == compare_variant].copy()
        comp_rows = comp_rows.merge(base_group_by_loc, on="location_id", how="inner")
        comp_rows = comp_rows[comp_rows["group_key"] == comp_rows["base_group"]].copy()
    
    if verbose:
        base_groups = base_rows["group_key"].dropna().unique().tolist()
        comp_groups = comp_rows["group_key"].dropna().unique().tolist()
        missing_groups = [g for g in base_groups if g not in set(comp_groups)]
        print(
            f"[f27] {baseline_variant} vs {compare_variant} | "
            f"base_rows={len(base_rows):,} comp_rows={len(comp_rows):,} "
            f"base_groups={len(base_groups):,} comp_groups={len(comp_groups):,} "
            f"missing_groups={len(missing_groups):,} fallback={used_fallback}"
        )
        if missing_groups:
            print(f"[f27] missing group_key sample: {missing_groups[:5]}")

    rel_map_base = base_rows[["location_id", "rel_path"]].assign(role="base")
    rel_map_comp = comp_rows[["rel_path"]].assign(
        location_id=comp_rows["base_location_id"], role="comp"
    )
    rel_map = pd.concat([rel_map_base, rel_map_comp], ignore_index=True).dropna()
    if rel_map.empty:
        return pd.DataFrame()

    ds = daily_stats.merge(rel_map, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()

    col = "DBT_max" if use_max else "DBT_mean"

    def agg_func(s: pd.Series) -> float:
        if use_max:
            return float(s.quantile(p))
        return float(s.mean())

    monthly = (
        ds.groupby(["location_id", "role", "month"], as_index=False)
        .agg(val=(col, agg_func))
    )

    # Pivot to base/comp and compute delta
    mon_p = monthly.pivot_table(index=["location_id", "month"], columns="role", values="val").reset_index()
    
    # Check if required columns exist
    if "comp" not in mon_p.columns or "base" not in mon_p.columns:
        return pd.DataFrame()
    
    # Drop months without both base+comp for this location
    mon_p = mon_p.dropna(subset=["base", "comp"])
    if mon_p.empty:
        return pd.DataFrame()

    mon_p["delta"] = mon_p["comp"] - mon_p["base"]

    max_p = mon_p.groupby("location_id", as_index=False)["delta"].max()
    max_p = max_p.dropna(subset=["delta"])
    if max_p.empty:
        return pd.DataFrame()

    # Metadata (robust to lat/lon column naming)
    base_rows = base_rows.copy()
    if "latitude" not in base_rows.columns and "lat" in base_rows.columns:
        base_rows["latitude"] = pd.to_numeric(base_rows["lat"], errors="coerce")
    if "longitude" not in base_rows.columns and "lon" in base_rows.columns:
        base_rows["longitude"] = pd.to_numeric(base_rows["lon"], errors="coerce")
    if "latitude" not in base_rows.columns:
        base_rows["latitude"] = pd.NA
    if "longitude" not in base_rows.columns:
        base_rows["longitude"] = pd.NA
    if "location_name" not in base_rows.columns:
        base_rows["location_name"] = base_rows.get("filename", "").astype(str)

    meta = base_rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )


    # Build wide table
    out = meta.merge(max_p.rename(columns={"delta": "max yearly"}), on="location_id", how="inner")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        col_name = month_labels[m - 1]
        vals = mon_p[mon_p["month"] == m][["location_id", "delta"]]
        out = out.merge(vals.rename(columns={"delta": col_name}), on="location_id", how="left")

    return out.sort_values("location_name")


def _ensure_hourly_has_month(hourly: pd.DataFrame) -> pd.DataFrame:
    """Ensure hourly DataFrame has a 'month' column (1-12) from datetime index or column."""
    if hourly is None or hourly.empty:
        return hourly
    df = hourly.copy()
    if "month" in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        df["month"] = df.index.month
        return df
    if "datetime" in df.columns:
        df["month"] = pd.to_datetime(df["datetime"], errors="coerce").dt.month
        return df
    # try index name
    if hasattr(df.index, "name") and df.index.name == "datetime":
        df = df.reset_index()
        df["month"] = pd.to_datetime(df["datetime"], errors="coerce").dt.month
        return df
    return df


def f124h__build_monthly_delta_table_from_hourly(
    hourly: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    baseline_variant: str,
    compare_variant: str,
    percentile: float,
    metric_key: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build monthly delta table from hourly DBT (robust: percentiles from hourly, not daily).

    Same output shape as f124. Expects hourly with DBT, rel_path; month is derived from datetime if missing.
    """
    p = max(0.0, min(1.0, float(percentile)))
    use_max = metric_key == "dTmax"
    if hourly is None or hourly.empty or idx is None or idx.empty:
        return pd.DataFrame()
    hourly = _ensure_hourly_has_month(hourly)
    if "month" not in hourly.columns:
        return pd.DataFrame()
    idx_no_cti = idx[~_bool_is_cti(idx)].copy()
    if "scenario" in hourly.columns:
        hourly = hourly[~hourly["scenario"].astype(str).str.contains("cti", case=False, na=False)].copy()
    base_rows = idx_no_cti[idx_no_cti["variant"] == baseline_variant].copy()
    if base_rows.empty:
        return pd.DataFrame()
    base_rows["base_group"] = base_rows["group_key"]
    base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
        base_location_id=("location_id", "first"),
        base_group=("group_key", "first"),
    )
    comp_rows = idx_no_cti[idx_no_cti["variant"] == compare_variant].copy()
    if comp_rows.empty:
        return pd.DataFrame()
    comp_rows = comp_rows.merge(base_group_by_key, on="group_key", how="inner")
    if comp_rows.empty:
        base_group_by_loc = base_rows.groupby("location_id", as_index=False).agg(
            base_location_id=("location_id", "first"),
            base_group=("group_key", "first"),
        )
        comp_rows = idx_no_cti[idx_no_cti["variant"] == compare_variant].copy()
        comp_rows = comp_rows.merge(base_group_by_loc, on="location_id", how="inner")
        comp_rows = comp_rows[comp_rows["group_key"] == comp_rows["base_group"]].copy()
    if comp_rows.empty:
        return pd.DataFrame()
    rel_map_base = base_rows[["location_id", "rel_path"]].assign(role="base")
    rel_map_comp = comp_rows[["rel_path"]].assign(
        location_id=comp_rows["base_location_id"], role="comp"
    )
    rel_map = pd.concat([rel_map_base, rel_map_comp], ignore_index=True).dropna()
    rel_map["rel_path"] = rel_map["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    df = hourly.copy()
    df["rel_path"] = df["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    ds = df.merge(rel_map, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()
    agg_func = (lambda s: float(s.quantile(p))) if use_max else (lambda s: float(s.mean()))
    monthly = ds.groupby(["location_id", "role", "month"], as_index=False).agg(val=("DBT", agg_func))
    mon_p = monthly.pivot_table(index=["location_id", "month"], columns="role", values="val").reset_index()
    if "comp" not in mon_p.columns or "base" not in mon_p.columns:
        return pd.DataFrame()
    mon_p = mon_p.dropna(subset=["base", "comp"])
    if mon_p.empty:
        return pd.DataFrame()
    mon_p["delta"] = mon_p["comp"] - mon_p["base"]
    max_p = mon_p.groupby("location_id", as_index=False)["delta"].max()
    max_p = max_p.dropna(subset=["delta"])
    if max_p.empty:
        return pd.DataFrame()
    base_rows = _ensure_lat_lon_columns(base_rows)
    if "location_name" not in base_rows.columns:
        base_rows["location_name"] = base_rows.get("filename", "").astype(str)
    meta = base_rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    out = meta.merge(max_p.rename(columns={"delta": "max yearly"}), on="location_id", how="inner")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        col_name = month_labels[m - 1]
        vals = mon_p[mon_p["month"] == m][["location_id", "delta"]]
        out = out.merge(vals.rename(columns={"delta": col_name}), on="location_id", how="left")
    return out.sort_values("location_name")


def f125h__compute_location_deltas_from_hourly(
    hourly: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    baseline_variant: str,
    compare_variant: str,
    percentile: float,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute max monthly deltas per location from hourly DBT (robust: percentiles from hourly).

    Same output shape as f125. Expects hourly with DBT, rel_path; month derived from datetime if missing.
    """
    p = max(0.0, min(1.0, float(percentile)))
    if hourly is None or hourly.empty or idx is None or idx.empty:
        return pd.DataFrame()
    hourly = _ensure_hourly_has_month(hourly)
    if "month" not in hourly.columns:
        return pd.DataFrame()
    idx_no_cti = idx[~_bool_is_cti(idx)].copy()
    if "scenario" in hourly.columns:
        hourly = hourly[~hourly["scenario"].astype(str).str.contains("cti", case=False, na=False)].copy()
    base_rows = idx_no_cti[idx_no_cti["variant"] == baseline_variant].copy()
    comp_rows = idx_no_cti[idx_no_cti["variant"] == compare_variant].copy()
    if base_rows.empty or comp_rows.empty:
        return pd.DataFrame()
    if "group_key" not in base_rows.columns:
        base_rows["group_key"] = base_rows["filename"].apply(f107__group_key_from_filename)
    if "group_key" not in comp_rows.columns:
        comp_rows["group_key"] = comp_rows["filename"].apply(f107__group_key_from_filename)
    base_rows["base_group"] = base_rows["group_key"]
    base_group_by_loc = (
        base_rows.groupby("location_id", as_index=False)["base_group"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )
    comp_rows = comp_rows.merge(base_group_by_loc, on="location_id", how="inner")
    comp_rows = comp_rows[comp_rows["group_key"] == comp_rows["base_group"]].copy()
    if comp_rows.empty:
        return pd.DataFrame()
    rel_map = pd.concat(
        [
            base_rows[["location_id", "rel_path"]].assign(role="base"),
            comp_rows[["location_id", "rel_path"]].assign(role="comp"),
        ],
        ignore_index=True,
    ).dropna()
    rel_map["rel_path"] = rel_map["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    df = hourly.copy()
    df["rel_path"] = df["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    ds = df.merge(rel_map, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()
    monthly = ds.groupby(["location_id", "role", "month"], as_index=False).agg(
        Tmax=("DBT", lambda s: float(s.quantile(p))),
        Tavg=("DBT", lambda s: float(s.mean())),
    )
    mon_p = (
        monthly.pivot_table(
            index=["location_id", "month"],
            columns="role",
            values=["Tmax", "Tavg"],
            aggfunc="first",
        )
        .reset_index()
    )
    flat_cols = ["location_id", "month"]
    for a, b in mon_p.columns[2:]:
        flat_cols.append(f"{a}_{b}")
    mon_p.columns = flat_cols
    needed = {"Tmax_base", "Tmax_comp", "Tavg_base", "Tavg_comp"}
    if not needed.issubset(set(mon_p.columns)):
        return pd.DataFrame()
    mon_p["dTmax"] = mon_p["Tmax_comp"] - mon_p["Tmax_base"]
    mon_p["dTavg"] = mon_p["Tavg_comp"] - mon_p["Tavg_base"]
    max_p = mon_p.groupby("location_id", as_index=False).agg(
        dTmax=("dTmax", "max"),
        dTavg=("dTavg", "max"),
    )
    max_p = max_p.dropna(subset=["dTmax", "dTavg"], how="all")
    if max_p.empty:
        return pd.DataFrame()
    base_rows = _ensure_lat_lon_columns(base_rows)
    if "location_name" not in base_rows.columns:
        base_rows["location_name"] = base_rows.get("filename", "").astype(str)
    meta = base_rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    out = meta.merge(max_p, on="location_id", how="inner")
    out = out.dropna(subset=["dTmax", "dTavg"], how="all")
    if out[["latitude", "longitude"]].notna().any().any():
        out = out.dropna(subset=["latitude", "longitude"])
    return out


def f125__compute_location_deltas_from_daily(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    baseline_variant: str,
    compare_variant: str,
    percentile: float,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute max monthly deltas per location from daily aggregates.

    This is the *robust* pairing logic:
    - Exclude CTI via _bool_is_cti(idx) and scenario contains 'cti' in daily_stats.
    - Pair baseline↔compare by (location_id AND group_key), so TMYx folder identity is preserved.
    """
    p = max(0.0, min(1.0, float(percentile)))

    if idx is None or idx.empty or daily_stats is None or daily_stats.empty:
        return pd.DataFrame()

    # Exclude CTI
    idx_no_cti = idx[~_bool_is_cti(idx)].copy()
    if idx_no_cti.empty:
        return pd.DataFrame()

    if "scenario" in daily_stats.columns:
        daily_stats = daily_stats[
            ~daily_stats["scenario"].astype(str).str.contains("cti", case=False, na=False)
        ].copy()

    base_rows = idx_no_cti[idx_no_cti["variant"] == baseline_variant].copy()
    comp_rows = idx_no_cti[idx_no_cti["variant"] == compare_variant].copy()
    if base_rows.empty or comp_rows.empty:
        return pd.DataFrame()
    comp_rows_raw = comp_rows.copy()

    # Ensure group_key exists
    if "group_key" not in base_rows.columns:
        base_rows["group_key"] = base_rows["filename"].apply(f107__group_key_from_filename)
    if "group_key" not in comp_rows.columns:
        comp_rows["group_key"] = comp_rows["filename"].apply(f107__group_key_from_filename)

    # Baseline identity for each location_id is its group_key
    base_rows["base_group"] = base_rows["group_key"]
    base_group_by_loc = (
        base_rows.groupby("location_id", as_index=False)["base_group"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )

    # Keep only compare rows that match baseline locations AND same group_key
    comp_rows = comp_rows.merge(base_group_by_loc, on="location_id", how="inner")
    comp_rows = comp_rows[comp_rows["group_key"] == comp_rows["base_group"]].copy()
    if comp_rows.empty:
        return pd.DataFrame()
    
    if verbose:
        base_keys = base_rows[["location_id", "group_key"]].drop_duplicates()
        comp_keys = comp_rows[["location_id", "group_key"]].drop_duplicates()
        missing = base_keys.merge(comp_keys, on=["location_id", "group_key"], how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"][["location_id", "group_key"]]
        base_groups = base_rows["group_key"].dropna().unique().tolist()
        comp_groups = comp_rows["group_key"].dropna().unique().tolist()
        missing_groups = [g for g in base_groups if g not in set(comp_groups)]
        print(
            f"[f28] {baseline_variant} vs {compare_variant} | "
            f"base_rows={len(base_rows):,} comp_rows_raw={len(comp_rows_raw):,} "
            f"comp_rows_paired={len(comp_rows):,} "
            f"base_keys={len(base_keys):,} matched_keys={len(comp_keys):,} "
            f"missing_keys={len(missing):,} "
            f"missing_groups={len(missing_groups):,}"
        )
        if not missing.empty:
            print(f"[f28] missing key sample: {missing.head(5).values.tolist()}")
        if missing_groups:
            print(f"[f28] missing group_key sample: {missing_groups[:5]}")

    # Build rel_path map for joining daily_stats
    rel_map = pd.concat(
        [
            base_rows[["location_id", "rel_path"]].assign(role="base"),
            comp_rows[["location_id", "rel_path"]].assign(role="comp"),
        ],
        ignore_index=True,
    ).dropna()
    if rel_map.empty:
        return pd.DataFrame()

    # Normalize rel_path
    rel_map["rel_path"] = rel_map["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    ds = daily_stats.copy()
    ds["rel_path"] = ds["rel_path"].astype(str).str.replace("\\", "/", regex=False)

    ds = ds.merge(rel_map, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()

    # Monthly stats for base/comp
    monthly = ds.groupby(["location_id", "role", "month"], as_index=False).agg(
        Tmax=("DBT_max", lambda s: float(s.quantile(p))),
        Tavg=("DBT_mean", lambda s: float(s.mean())),
    )

    mon_p = (
        monthly.pivot_table(
            index=["location_id", "month"],
            columns="role",
            values=["Tmax", "Tavg"],
            aggfunc="first",
        )
        .reset_index()
    )

    # Flatten columns
    if mon_p.shape[1] < 6:
        return pd.DataFrame()

    flat_cols = ["location_id", "month"]
    for a, b in mon_p.columns[2:]:
        flat_cols.append(f"{a}_{b}")
    mon_p.columns = flat_cols

    needed = {"Tmax_base", "Tmax_comp", "Tavg_base", "Tavg_comp"}
    if not needed.issubset(set(mon_p.columns)):
        return pd.DataFrame()

    mon_p["dTmax"] = mon_p["Tmax_comp"] - mon_p["Tmax_base"]
    mon_p["dTavg"] = mon_p["Tavg_comp"] - mon_p["Tavg_base"]

    max_p = mon_p.groupby("location_id", as_index=False).agg(
        dTmax=("dTmax", "max"),
        dTavg=("dTavg", "max"),
    )
    max_p = max_p.dropna(subset=["dTmax", "dTavg"], how="all")
    if max_p.empty:
        return pd.DataFrame()

    # Metadata from baseline
    base_rows = _ensure_lat_lon_columns(base_rows)
    if "location_name" not in base_rows.columns:
        base_rows["location_name"] = base_rows.get("filename", "").astype(str)

    meta = base_rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )

    out = meta.merge(max_p, on="location_id", how="inner")
    out = out.dropna(subset=["dTmax", "dTavg"], how="all")
    if out[["latitude", "longitude"]].notna().any().any():
        out = out.dropna(subset=["latitude", "longitude"])
    return out


def f28bh__compute_location_stats_for_variant_from_hourly(
    hourly: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    variant: str,
    percentile: float,
) -> pd.DataFrame:
    """
    Compute absolute annual stats per location from hourly DBT (robust: percentiles from hourly).

    - Tmax: percentile of all hourly DBT (qP) per location
    - Tavg: mean of all hourly DBT per location

    Expects hourly with columns: DBT, rel_path. Index must have rel_path, variant, location_id.
    CTI is excluded (variant == "cti" or is_cti in index).
    """
    if hourly is None or hourly.empty or idx is None or idx.empty or variant == "cti":
        return pd.DataFrame()
    idx_epw = idx[~_bool_is_cti(idx)].copy()
    rows = idx_epw[idx_epw["variant"] == variant].copy()
    if rows.empty:
        return pd.DataFrame()
    rels = rows[["location_id", "rel_path"]].dropna()
    rels["rel_path"] = rels["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    df = hourly.copy()
    if "DBT" not in df.columns or "rel_path" not in df.columns:
        return pd.DataFrame()
    df["rel_path"] = df["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    ds = df.merge(rels, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()
    p = max(0.0, min(1.0, float(percentile)))
    annual = ds.groupby("location_id", as_index=False).agg(
        Tmax=("DBT", lambda s: s.quantile(p)),
        Tavg=("DBT", "mean"),
    )
    rows = _ensure_lat_lon_columns(rows)
    if "location_name" not in rows.columns:
        rows["location_name"] = pd.NA
    meta = rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    out = meta.merge(annual, on="location_id", how="inner")
    if out[["latitude", "longitude"]].notna().any().any():
        out = out.dropna(subset=["latitude", "longitude"])
    return out


def f28b__compute_location_stats_for_variant_from_daily(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    variant: str,
    percentile: float,
) -> pd.DataFrame:
    """
    Compute absolute annual stats per location (single variant) from daily aggregates.

    - Tmax: percentile of daily max DBT (qP)
    - Tavg: mean of daily mean DBT

    CTI is excluded from comparison logic — this is for TMYx/RCP only.
    """
    if variant == "cti":
        return pd.DataFrame()
    p = max(0.0, min(1.0, float(percentile)))

    idx_epw = idx[~idx["is_cti"]].copy() if "is_cti" in idx.columns else idx.copy()
    if "scenario" in daily_stats.columns:
        daily_stats = daily_stats[~daily_stats["scenario"].astype(str).str.contains("cti", case=False, na=False)].copy()

    rows = idx_epw[idx_epw["variant"] == variant].copy()
    if rows.empty:
        return pd.DataFrame()

    rels = rows[["location_id", "rel_path"]].dropna()
    if rels.empty:
        return pd.DataFrame()

    ds = daily_stats.merge(rels, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()

    annual = ds.groupby(["location_id"], as_index=False).agg(
        Tmax=("DBT_max", lambda s: s.quantile(p)),
        Tavg=("DBT_mean", "mean"),
    )




    # ---- Robust coords: support both (latitude/longitude) and (lat/lon) ----
    rows = rows.copy()

    # Ensure location_name exists
    if "location_name" not in rows.columns:
        rows["location_name"] = pd.NA

    # Map lat/lon -> latitude/longitude if needed
    if "latitude" not in rows.columns and "lat" in rows.columns:
        rows["latitude"] = pd.to_numeric(rows["lat"], errors="coerce")
    if "longitude" not in rows.columns and "lon" in rows.columns:
        rows["longitude"] = pd.to_numeric(rows["lon"], errors="coerce")

    # Ensure columns exist (avoid KeyError)
    if "latitude" not in rows.columns:
        rows["latitude"] = pd.NA
    if "longitude" not in rows.columns:
        rows["longitude"] = pd.NA

    meta = rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )





    out = meta.merge(annual, on="location_id", how="inner")
    if out[["latitude", "longitude"]].notna().any().any():
        out = out.dropna(subset=["latitude", "longitude"])
    return out


def f28c__compute_location_stats_cti_from_daily(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    variant: str = "cti",
    percentile: float,
) -> pd.DataFrame:
    """
    Compute absolute annual stats per location (CTI) from daily aggregates.

    - Tmax: percentile of daily max DBT (qP)
    - Tavg: mean of daily mean DBT
    """
    if daily_stats is None or daily_stats.empty or idx is None or idx.empty:
        return pd.DataFrame()
    p = max(0.0, min(1.0, float(percentile)))

    rows = idx[idx["variant"] == variant].copy()
    if rows.empty:
        return pd.DataFrame()

    rels = rows[["location_id", "rel_path"]].dropna().drop_duplicates()
    if rels.empty:
        return pd.DataFrame()

    ds = daily_stats.merge(rels, on="rel_path", how="inner")
    if ds.empty:
        return pd.DataFrame()

    annual = ds.groupby(["location_id"], as_index=False).agg(
        Tmax=("DBT_max", lambda s: s.quantile(p)),
        Tavg=("DBT_mean", "mean"),
    )

    rows = _ensure_lat_lon_columns(rows)
    if "location_name" not in rows.columns:
        rows["location_name"] = pd.NA

    meta = rows.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )

    out = meta.merge(annual, on="location_id", how="inner")
    if out[["latitude", "longitude"]].notna().any().any():
        out = out.dropna(subset=["latitude", "longitude"])
    return out


def f128__cti_compute_location_stats(
    path_dbt_csv: Path,
    path_list_csv: Path,
    *,
    percentile: float,
) -> pd.DataFrame:
    """
    Compute annual Tmax (percentile of daily max) and Tavg (mean of daily mean) per location
    from CTI (Itaca) weather station CSVs.

    Expected CSVs:
    - path_list_csv: location_id, location_name, latitude, longitude, region (or station_id etc.)
    - path_dbt_csv: location_id (or station_id), datetime (or date + hour), DBT
    Marker positions (latitude, longitude) come from the list file; temperature stats (Tmax, Tavg)
    are computed from hourly DBT the same way as TMYx (percentile of daily max, mean of daily mean).
    Region can come from a column or be derived from location_id when format is REG__... (e.g. AB__AQ__L'Aquila -> AB).
    """
    if not path_list_csv.exists() or not path_dbt_csv.exists():
        return pd.DataFrame()
    list_df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            list_df = pd.read_csv(path_list_csv, encoding=encoding)
            if len(list_df.columns) == 1:
                list_df = pd.read_csv(path_list_csv, sep=";", encoding=encoding)
            break
        except Exception:
            continue
    if list_df is None or list_df.empty:
        return pd.DataFrame()
    dbt_df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            dbt_df = pd.read_csv(path_dbt_csv, encoding=encoding, low_memory=False)
            if len(dbt_df.columns) == 1:
                dbt_df = pd.read_csv(path_dbt_csv, sep=";", encoding=encoding, low_memory=False)
            break
        except Exception:
            continue
    if dbt_df is None or dbt_df.empty:
        return pd.DataFrame()
    # Normalize column names: lower, strip, and "space" -> "underscore" so "Location ID" matches "location_id"
    list_cols = {}
    for c in list_df.columns:
        k = str(c).lower().strip()
        list_cols[k] = c
        list_cols[k.replace(" ", "_")] = c
    def _list_get(*keys):
        for k in keys:
            if k in list_cols:
                return list_cols[k]
        return None
    id_col_list = _list_get("location_id", "station_id", "id", "station_code", "code", "stazione", "location id", "station id")
    name_col = _list_get("location_name", "station_name", "name", "nome", "station", "denominazione", "location name", "station name")
    lat_col = _list_get("latitude", "lat", "latitudine", "y")
    lon_col = _list_get("longitude", "lon", "lng", "longitudine", "x")
    region_col = _list_get("region", "reg_shortname", "regione", "reg", "region_code", "codice_regione", "reg shortname")
    if not all([id_col_list, name_col, lat_col, lon_col]):
        return pd.DataFrame()
    list_df = list_df.rename(columns={
        id_col_list: "location_id",
        name_col: "location_name",
        lat_col: "latitude",
        lon_col: "longitude",
    })
    if region_col:
        list_df["region"] = list_df[region_col].astype(str).str.strip()
    else:
        list_df["region"] = "CTI"
    # Derive region from location_id when format is REG__... (e.g. AB__AQ__L'Aquila -> AB)
    def _region_from_location_id(loc: str):
        s = str(loc).strip()
        if "__" in s:
            part = s.split("__")[0]
            if len(part) == 2 and part.isalpha():
                return part.upper()
        return None
    derived = list_df["location_id"].astype(str).apply(_region_from_location_id)
    list_df["region"] = derived.fillna(list_df["region"])
    list_df["location_id"] = list_df["location_id"].astype(str)
    for col in ("latitude", "longitude"):
        if list_df[col].dtype.kind in ("O", "S", "U") or str(list_df[col].dtype) == "object":
            list_df[col] = list_df[col].astype(str).str.replace(",", ".", regex=False)
        list_df[col] = pd.to_numeric(list_df[col], errors="coerce")
    list_df = list_df.dropna(subset=["latitude", "longitude"])
    list_df = list_df[["location_id", "location_name", "latitude", "longitude", "region"]].drop_duplicates()

    dbt_cols = {c.lower().strip(): c for c in dbt_df.columns}
    id_col_dbt = (
        dbt_cols.get("location_id") or dbt_cols.get("station_id") or dbt_cols.get("id")
        or dbt_cols.get("station_code") or dbt_cols.get("code") or dbt_cols.get("stazione")
        or dbt_cols.get("cod_stazione") or dbt_cols.get("station") or dbt_cols.get("site_id")
        or dbt_cols.get("location") or dbt_cols.get("point_id")
    )
    dt_col = (
        dbt_cols.get("datetime") or dbt_cols.get("date") or dbt_cols.get("timestamp")
        or dbt_cols.get("time") or dbt_cols.get("data") or dbt_cols.get("dataora")
    )
    dbt_val_col = (
        dbt_cols.get("dbt") or dbt_cols.get("dry_bulb") or dbt_cols.get("temperature")
        or dbt_cols.get("temp") or dbt_cols.get("t") or dbt_cols.get("t_air") or dbt_cols.get("temperatura")
    )
    if not all([id_col_dbt, dbt_val_col]):
        return pd.DataFrame()
    dbt_df = dbt_df.rename(columns={id_col_dbt: "location_id", dbt_val_col: "DBT"})
    if dt_col:
        dbt_df["datetime"] = pd.to_datetime(dbt_df[dt_col], errors="coerce")
    else:
        # Try date + hour
        date_c = dbt_cols.get("date")
        hour_c = dbt_cols.get("hour")
        if date_c and hour_c:
            dbt_df["datetime"] = pd.to_datetime(
                dbt_df[date_c].astype(str) + " " + dbt_df[hour_c].astype(str).str.zfill(2) + ":00:00",
                errors="coerce",
            )
        else:
            return pd.DataFrame()
    dbt_df = dbt_df.dropna(subset=["datetime", "DBT"])
    dbt_df["location_id"] = dbt_df["location_id"].astype(str)
    dbt_df["month"] = dbt_df["datetime"].dt.month
    dbt_df["day"] = dbt_df["datetime"].dt.day
    daily = (
        dbt_df.groupby(["location_id", "month", "day"], as_index=False)
        .agg(DBT_mean=("DBT", "mean"), DBT_max=("DBT", "max"))
    )
    p = max(0.0, min(1.0, float(percentile)))
    annual = (
        daily.groupby("location_id", as_index=False)
        .agg(
            Tmax=("DBT_max", lambda s: s.quantile(p)),
            Tavg=("DBT_mean", "mean"),
        )
    )
    out = list_df.merge(annual, on="location_id", how="inner")
    return out.dropna(subset=["latitude", "longitude"])


def f94b__cti_load_list_only(path_list_csv: Path) -> pd.DataFrame:
    """
    Load CTI station list (locations only) from the list CSV.
    Returns DataFrame with location_id, location_name, latitude, longitude, region.
    Used to show the CTI map when DBT stats are not available (list-only mode).
    Accepts many column name variants (English, Italian, common abbreviations).
    Tries multiple encodings (utf-8-sig, utf-8, latin-1, cp1252).
    """
    if not path_list_csv.exists():
        return pd.DataFrame()
    list_df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            list_df = pd.read_csv(path_list_csv, encoding=encoding)
            break
        except Exception:
            continue
    if list_df is None or list_df.empty:
        return pd.DataFrame()
    # If only one column, try semicolon delimiter (common in European CSVs)
    if len(list_df.columns) == 1:
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                list_df = pd.read_csv(path_list_csv, sep=";", encoding=enc)
                if len(list_df.columns) > 1:
                    break
            except Exception:
                continue
    if list_df.empty or len(list_df.columns) < 2:
        return pd.DataFrame()
    # Normalize column lookup: lower, strip, and also "space" -> "underscore" so "Location ID" matches "location_id"
    list_cols = {}
    for c in list_df.columns:
        k = str(c).lower().strip()
        list_cols[k] = c
        list_cols[k.replace(" ", "_")] = c
    def _get(*keys):
        for k in keys:
            if k in list_cols:
                return list_cols[k]
        return None
    id_col = _get("location_id", "station_id", "id", "station_code", "code", "stazione", "location id", "station id")
    name_col = _get(
        "location_name", "station_name", "name", "nome", "station", "denominazione",
        "location name", "station name", "location"
    )
    lat_col = _get("latitude", "lat", "latitudine", "y", "lat.")
    lon_col = _get("longitude", "lon", "lng", "longitudine", "x", "lon.", "lng.")
    region_col = _get("region", "reg_shortname", "regione", "reg", "region_code", "codice_regione", "reg shortname")
    if not all([name_col, lat_col, lon_col]):
        return pd.DataFrame()
    list_df = list_df.rename(columns={
        name_col: "location_name",
        lat_col: "latitude",
        lon_col: "longitude",
    })
    if id_col:
        list_df = list_df.rename(columns={id_col: "location_id"})
    else:
        list_df["location_id"] = list_df["location_name"]
    if region_col:
        list_df["region"] = list_df[region_col].astype(str).str.strip()
    else:
        list_df["region"] = "CTI"
    # Derive region from location_id when format is REG__... (e.g. AB__AQ__L'Aquila -> AB)
    def _region_from_loc(loc):
        s = str(loc).strip()
        if "__" in s:
            part = s.split("__")[0]
            if len(part) == 2 and part.isalpha():
                return part.upper()
        return None
    derived = list_df["location_id"].astype(str).apply(_region_from_loc)
    list_df["region"] = derived.fillna(list_df["region"])
    list_df["location_id"] = list_df["location_id"].astype(str)
    list_df["location_name"] = list_df["location_name"].astype(str)

    def _normalize_loc_id(value: str) -> str:
        s = str(value or "").strip()
        s = re.sub(r"[\\s]+", "_", s)
        s = re.sub(r"[\\'\\\"]+", "", s)
        s = re.sub(r"[^A-Za-z0-9_\\.-]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    list_df["location_id"] = list_df["location_id"].apply(_normalize_loc_id)
    list_df["location_name"] = list_df["location_name"].str.strip()
    # Coerce lat/lon to numeric (allow comma as decimal separator, e.g. "41,92" -> 41.92)
    for col in ["latitude", "longitude"]:
        if list_df[col].dtype.kind in ("O", "S", "U") or str(list_df[col].dtype) == "object":
            list_df[col] = list_df[col].astype(str).str.replace(",", ".", regex=False)
        list_df[col] = pd.to_numeric(list_df[col], errors="coerce")
    out = list_df[["location_id", "location_name", "latitude", "longitude", "region"]].drop_duplicates()
    out = out.dropna(subset=["latitude", "longitude"])
    return out


def f94c__cti_dbt_location_counts(path_dbt_csv: Path) -> pd.Series:
    """
    Load the CTI DBT (hourly temperature) CSV and return hourly record count per location_id.
    Returns Series with index=location_id (str), value=number of hourly rows.
    Used to check which locations have hourly temperature data.
    Accepts many column name variants for the location identifier.
    """
    if not path_dbt_csv.exists():
        return pd.Series(dtype=int)
    dbt_df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            dbt_df = pd.read_csv(path_dbt_csv, encoding=encoding, low_memory=False)
            break
        except Exception:
            continue
    if dbt_df is None or dbt_df.empty:
        return pd.Series(dtype=int)
    # If only one column, try semicolon
    if len(dbt_df.columns) == 1:
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                dbt_df = pd.read_csv(path_dbt_csv, sep=";", encoding=enc, low_memory=False)
                if len(dbt_df.columns) > 1:
                    break
            except Exception:
                continue
    if dbt_df.empty or len(dbt_df.columns) < 2:
        return pd.Series(dtype=int)
    dbt_cols = {c.lower().strip(): c for c in dbt_df.columns}
    id_col = (
        dbt_cols.get("location_id") or dbt_cols.get("station_id") or dbt_cols.get("id")
        or dbt_cols.get("station_code") or dbt_cols.get("code") or dbt_cols.get("stazione")
        or dbt_cols.get("cod_stazione") or dbt_cols.get("station") or dbt_cols.get("site_id")
        or dbt_cols.get("location") or dbt_cols.get("point_id")
    )
    if not id_col:
        return pd.Series(dtype=int)
    dbt_df = dbt_df.rename(columns={id_col: "location_id"})
    dbt_df["location_id"] = dbt_df["location_id"].astype(str)
    return dbt_df.groupby("location_id").size()


def f129__cti_build_daily_profiles_bundle(
    path_dbt_csv: Path,
    path_list_csv: Path,
    *,
    daily_stat: str = "max",
) -> Dict[str, Any]:
    """
    Build daily DBT profiles per location from CTI CSVs for D3 marker-click charts.
    Returns same shape as f22c: {"keys": [[month, day], ...], "profiles": {loc_id: {"name": str, "series": [..]}}}.
    """
    if daily_stat not in ("mean", "max"):
        raise ValueError("daily_stat must be 'mean' or 'max'")
    if not path_list_csv.exists() or not path_dbt_csv.exists():
        return {"keys": [], "profiles": {}}
    try:
        list_df = pd.read_csv(path_list_csv)
        dbt_df = pd.read_csv(path_dbt_csv)
    except Exception:
        return {"keys": [], "profiles": {}}
    list_cols = {c.lower(): c for c in list_df.columns}
    id_col_list = list_cols.get("location_id") or list_cols.get("station_id") or list_cols.get("id")
    name_col = list_cols.get("location_name") or list_cols.get("station_name") or list_cols.get("name")
    if not id_col_list or not name_col:
        return {"keys": [], "profiles": {}}
    list_df = list_df.rename(columns={id_col_list: "location_id", name_col: "location_name"})
    list_df["location_id"] = list_df["location_id"].astype(str)
    name_by_loc = list_df.set_index("location_id")["location_name"].to_dict()

    dbt_cols = {c.lower(): c for c in dbt_df.columns}
    id_col_dbt = dbt_cols.get("location_id") or dbt_cols.get("station_id") or dbt_cols.get("id")
    dt_col = dbt_cols.get("datetime") or dbt_cols.get("date") or dbt_cols.get("timestamp")
    dbt_val_col = dbt_cols.get("dbt") or dbt_cols.get("dry_bulb") or dbt_cols.get("temperature")
    if not all([id_col_dbt, dbt_val_col]):
        return {"keys": [], "profiles": {}}
    dbt_df = dbt_df.rename(columns={id_col_dbt: "location_id", dbt_val_col: "DBT"})
    if dt_col:
        dbt_df["datetime"] = pd.to_datetime(dbt_df[dt_col], errors="coerce")
    else:
        date_c, hour_c = dbt_cols.get("date"), dbt_cols.get("hour")
        if date_c and hour_c:
            dbt_df["datetime"] = pd.to_datetime(
                dbt_df[date_c].astype(str) + " " + dbt_df[hour_c].astype(str).str.zfill(2) + ":00:00",
                errors="coerce",
            )
        else:
            return {"keys": [], "profiles": {}}
    dbt_df = dbt_df.dropna(subset=["datetime", "DBT"])
    dbt_df["location_id"] = dbt_df["location_id"].astype(str)
    dbt_df["month"] = dbt_df["datetime"].dt.month
    dbt_df["day"] = dbt_df["datetime"].dt.day
    if daily_stat == "mean":
        daily = (
            dbt_df.groupby(["location_id", "month", "day"], as_index=False)
            .agg(DBT=("DBT", "mean"))
        )
    else:
        daily = (
            dbt_df.groupby(["location_id", "month", "day"], as_index=False)
            .agg(DBT=("DBT", "max"))
        )
    base_year = 2001
    all_days = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    keys = [(int(d.month), int(d.day)) for d in all_days]
    key_index = {k: i for i, k in enumerate(keys)}
    locs = daily["location_id"].unique().tolist()
    profiles: Dict[str, Any] = {}
    for loc in locs:
        profiles[loc] = {"name": str(name_by_loc.get(loc, loc)), "series": [None] * len(keys)}
    for _, r in daily.iterrows():
        loc = str(r["location_id"])
        k = (int(r["month"]), int(r["day"]))
        j = key_index.get(k)
        if j is None or loc not in profiles:
            continue
        v = None if pd.isna(r["DBT"]) else float(round(r["DBT"], 3))
        profiles[loc]["series"][j] = v
    return {"keys": [[m, d] for (m, d) in keys], "profiles": profiles}


def f114__compute_location_deltas(
    idx: pd.DataFrame,
    file_stats: pd.DataFrame,
    baseline_variant: str,
    compare_variant: str,
) -> pd.DataFrame:
    meta = idx.merge(file_stats, on="rel_path", how="inner")

    base = meta[meta["variant"] == baseline_variant].copy()
    comp = meta[meta["variant"] == compare_variant].copy()

    base_loc = base.groupby("location_id", as_index=False).agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        Tmax_base=("Tmax", "mean"),
        Tavg_base=("Tavg", "mean"),
    )
    comp_loc = comp.groupby("location_id", as_index=False).agg(
        Tmax_comp=("Tmax", "mean"),
        Tavg_comp=("Tavg", "mean"),
    )

    out = base_loc.merge(comp_loc, on="location_id", how="inner")
    out["dTmax"] = out["Tmax_comp"] - out["Tmax_base"]
    out["dTavg"] = out["Tavg_comp"] - out["Tavg_base"]
    out = out.dropna(subset=["latitude", "longitude"])
    return out


def f115__load_all_data_with_progress(
    index_path: Path,
    tidy_path: Path,
    *,
    show_details: bool = False,
    compute_file_stats: bool = True,
):
    """
    Load/cached-read all datasets.

    - show_details=False: silent (no top-of-page expander)
    - show_details=True: shows a progress/status UI
    """
    if not show_details:
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not tidy_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {tidy_path}")

        idx = f110__load_index(index_path)
        tidy_raw = f111__load_tidy_parquet_raw(tidy_path)
        tidy = f112__postprocess_tidy_parquet(tidy_raw)
        file_stats = f113__build_file_stats(tidy) if compute_file_stats else None
        return idx, tidy_raw, tidy, file_stats

    status = st.status("Loading data…", expanded=True)
    bar = st.progress(0, text="Starting…")

    def step(pct: int, msg: str):
        bar.progress(int(pct), text=msg)
        status.update(label=msg)

    try:
        step(5, "Validating paths…")
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not tidy_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {tidy_path}")

        step(20, "Reading + caching index…")
        idx = f110__load_index(index_path)
        status.write(f"Index: {len(idx):,} files/records")

        step(55, "Reading + caching parquet…")
        tidy_raw = f111__load_tidy_parquet_raw(tidy_path)
        status.write(f"Parquet raw: {len(tidy_raw):,} rows × {len(tidy_raw.columns)} columns")

        step(75, "Post-processing parquet…")
        tidy = f112__postprocess_tidy_parquet(tidy_raw)
        status.write(f"Parquet cleaned: {len(tidy):,} rows")

        step(90, "Computing + caching file stats…")
        if compute_file_stats:
            file_stats = f113__build_file_stats(tidy)
            status.write(f"File stats: {len(file_stats):,} EPW files")
        else:
            file_stats = None

        step(100, "Done")
        status.update(label="Data loaded", state="complete")
        return idx, tidy_raw, tidy, file_stats
    except Exception:
        status.update(label="Loading failed", state="error")
        raise
    finally:
        bar.empty()


# -----------------------------
# Selection-aware time series
# -----------------------------
def f116__baseline_group_for_location(idx: pd.DataFrame, location_id: str, baseline_variant: str) -> Optional[str]:
    """Prefer group_key (folder name / baseline stem) for pairing baseline ↔ compare."""
    sub = idx[(idx["location_id"] == location_id) & (idx["variant"] == baseline_variant)]
    if sub.empty:
        return None
    g = sub.iloc[0].get("group_key")
    if g is not None and str(g).strip():
        return str(g)
    fn = sub.iloc[0].get("filename")
    if isinstance(fn, str) and fn:
        return f107__group_key_from_filename(fn)
    return None


@st.cache_data(show_spinner=False)
def f117__location_timeseries_for_variant(
    tidy: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_id: str,
    variant: str,
    group_key: Optional[str] = None,
) -> pd.DataFrame:
    q = (idx["location_id"] == location_id) & (idx["variant"] == variant)
    if group_key:
        q = q & (idx["group_key"] == group_key)
    rel_paths = idx.loc[q, "rel_path"]
    rel_paths = [p for p in rel_paths.dropna().unique().tolist() if isinstance(p, str)]
    if not rel_paths:
        return pd.DataFrame(columns=["datetime", "DBT", "RH"])

    sub = tidy[tidy["rel_path"].isin(rel_paths)][["datetime", "DBT", "RH"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["datetime", "DBT", "RH"])
    return (
        sub.groupby("datetime", as_index=False)
        .agg(DBT=("DBT", "mean"), RH=("RH", "mean"))
        .sort_values("datetime")
    )


def f118__build_location_scatter_df(
    tidy: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_id: str,
    baseline_variant: str,
    compare_variant: str,
) -> pd.DataFrame:
    base_group = f116__baseline_group_for_location(idx, location_id, baseline_variant)

    base = f117__location_timeseries_for_variant(
        tidy, idx, location_id=location_id, variant=baseline_variant, group_key=base_group
    ).rename(columns={"DBT": "DBT_base", "RH": "RH_base"})

    comp = f117__location_timeseries_for_variant(
        tidy, idx, location_id=location_id, variant=compare_variant, group_key=base_group
    ).rename(columns={"DBT": "DBT_comp", "RH": "RH_comp"})

    if base.empty or comp.empty:
        return pd.DataFrame(columns=["month", "day", "hour", "DBT_base", "DBT_comp", "RH_base", "RH_comp"])

    for df in (base, comp):
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["hour"] = df["datetime"].dt.hour

    base_g = base.groupby(["month", "day", "hour"], as_index=False).agg(
        DBT_base=("DBT_base", "mean"),
        RH_base=("RH_base", "mean"),
    )
    comp_g = comp.groupby(["month", "day", "hour"], as_index=False).agg(
        DBT_comp=("DBT_comp", "mean"),
        RH_comp=("RH_comp", "mean"),
    )

    merged = base_g.merge(comp_g, on=["month", "day", "hour"], how="inner")
    return merged


# -----------------------------
# Fast per-click charts: pre-aggregated daily stats
# -----------------------------
@st.cache_data(show_spinner=False)
def f121__daily_stats_by_rel_path(tidy_parquet_path: str) -> pd.DataFrame:
    """
    Pre-aggregate the huge hourly parquet into a compact daily table.

    Output columns:
      rel_path, month, day, DBT_mean, DBT_max
    """
    tidy = pd.read_parquet(tidy_parquet_path, columns=["rel_path", "datetime", "DBT"])
    tidy["datetime"] = pd.to_datetime(tidy["datetime"], errors="coerce")
    tidy = tidy.dropna(subset=["datetime"])
    tidy["month"] = tidy["datetime"].dt.month.astype(int)
    tidy["day"] = tidy["datetime"].dt.day.astype(int)

    out = (
        tidy.groupby(["rel_path", "month", "day"], as_index=False)
        .agg(DBT_mean=("DBT", "mean"), DBT_max=("DBT", "max"))
    )
    return out


def f122__build_location_daily_join(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_id: str,
    baseline_variant: str,
    compare_variant: str,
    stat: str,
) -> pd.DataFrame:
    """
    Fast per-location daily join built from `f121__daily_stats_by_rel_path`.

    - Matches compare files to the baseline group (TMYx vs TMYx.2007-2021 etc)
    - Aligns by month/day (typical year)

    Returns columns:
      month, day, date, DBT_base, DBT_comp, delta
    """
    if stat not in ("mean", "max"):
        raise ValueError("stat must be 'mean' or 'max'")

    base_group = f116__baseline_group_for_location(idx, location_id, baseline_variant)
    if not base_group:
        return pd.DataFrame(columns=["month", "day", "date", "DBT_base", "DBT_comp", "delta"])

    def rels_for(variant: str) -> list[str]:
        q = (idx["location_id"] == location_id) & (idx["variant"] == variant) & (idx["group_key"] == base_group)
        return [p for p in idx.loc[q, "rel_path"].dropna().unique().tolist() if isinstance(p, str)]

    base_rels = rels_for(baseline_variant)
    comp_rels = rels_for(compare_variant)
    if not base_rels or not comp_rels:
        return pd.DataFrame(columns=["month", "day", "date", "DBT_base", "DBT_comp", "delta"])

    col = "DBT_max" if stat == "max" else "DBT_mean"

    base = (
        daily_stats[daily_stats["rel_path"].isin(base_rels)][["month", "day", col]]
        .groupby(["month", "day"], as_index=False)
        .agg(DBT_base=(col, "mean"))
    )
    comp = (
        daily_stats[daily_stats["rel_path"].isin(comp_rels)][["month", "day", col]]
        .groupby(["month", "day"], as_index=False)
        .agg(DBT_comp=(col, "mean"))
    )
    merged = base.merge(comp, on=["month", "day"], how="inner")
    merged["delta"] = merged["DBT_comp"] - merged["DBT_base"]
    # Build a synthetic date for plotting (non-leap)
    merged["date"] = pd.to_datetime(
        dict(year=2001, month=merged["month"].astype(int), day=merged["day"].astype(int)),
        errors="coerce",
    )
    return merged.sort_values(["month", "day"])


def f25b__build_location_daily_for_variant(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_id: str,
    variant: str,
    stat: str,
) -> pd.DataFrame:
    """
    Fast per-location daily series for a single variant, built from `f121__daily_stats_by_rel_path`.

    Returns columns:
      month, day, date, DBT
    """
    if stat not in ("mean", "max"):
        raise ValueError("stat must be 'mean' or 'max'")

    q = (idx["location_id"].astype(str) == str(location_id)) & (idx["variant"] == variant)
    rels = idx.loc[q, "rel_path"].dropna().unique().tolist()
    rels = [p for p in rels if isinstance(p, str)]
    if not rels:
        return pd.DataFrame(columns=["month", "day", "date", "DBT"])

    col = "DBT_max" if stat == "max" else "DBT_mean"
    base = (
        daily_stats[daily_stats["rel_path"].isin(rels)][["month", "day", col]]
        .groupby(["month", "day"], as_index=False)
        .agg(DBT=(col, "mean"))
    )
    base["date"] = pd.to_datetime(
        dict(year=2001, month=base["month"].astype(int), day=base["day"].astype(int)),
        errors="coerce",
    )
    base = base.dropna(subset=["date"])
    return base.sort_values(["month", "day"])


# -----------------------------
# Plotly/D3 charts (implemented in fn__libs_charts; re-exported here)
# -----------------------------
from libs.fn__libs_charts import (
    f201__plotly_italy_map as f19__plotly_italy_map,
    f202__plotly_italy_map_abs as f19b__plotly_italy_map_abs,
    f203__parse_plotly_selection as f20__parse_plotly_selection,
    f204__d3_dashboard_html_abs as f23b__d3_dashboard_html_abs,
    f205__d3_dashboard_html as f23__d3_dashboard_html,
    f206__d3_region_dashboard_html as f23c__d3_region_dashboard_html,
    f207__d3_region_maps_html as f23d__d3_region_maps_html,
    f208__plotly_tmyx_heatmap as f31__plotly_tmyx_heatmap,
    f209__plotly_tmyx_daily_range_scatter as f32__plotly_tmyx_daily_range_scatter,
    f210__plotly_tmyx_stacked_column as f33__plotly_tmyx_stacked_column,
    f211__plotly_tmyx_heatmap_subplots as f34__plotly_tmyx_heatmap_subplots,
    f212__plotly_tmyx_scatter_subplots as f35__plotly_tmyx_scatter_subplots,
    f213__plotly_tmyx_stacked_subplots as f36__plotly_tmyx_stacked_subplots,
)


# -----------------------------
# D3/JS embedded dashboard data
# -----------------------------
@st.cache_data(show_spinner=False)
def f119__build_month_hour_profiles(
    tidy: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_ids: Tuple[str, ...],
    baseline_variant: str,
    compare_variant: str,
) -> Dict[str, Any]:
    locs = [str(x) for x in location_ids]
    idx_sub = idx[idx["location_id"].isin(locs)].copy()
    if idx_sub.empty:
        return {"keys": [], "profiles": {}}

    base_rows = idx_sub[idx_sub["variant"] == baseline_variant].copy()
    if base_rows.empty:
        return {"keys": [], "profiles": {}}

    base_rows["base_group"] = base_rows["group_key"]
    base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
        base_location_id=("location_id", "first"),
        base_group=("group_key", "first"),
    )

    base_rels = base_rows[["location_id", "rel_path", "base_group"]].dropna().drop_duplicates()
    base_rels["role"] = "base"

    comp_rows = idx[idx["variant"] == compare_variant].copy()
    comp_rows = comp_rows.merge(base_group_by_key, on="group_key", how="inner")
    comp_rows = comp_rows[comp_rows["base_location_id"].isin(locs)]
    comp_rels = comp_rows[["rel_path"]].assign(
        location_id=comp_rows["base_location_id"], role="comp"
    ).dropna().drop_duplicates()

    rel_map = pd.concat([base_rels[["location_id", "rel_path", "role"]], comp_rels], ignore_index=True)
    if rel_map.empty:
        return {"keys": [], "profiles": {}}

    needed_rel_paths = rel_map["rel_path"].unique().tolist()
    tidy_sub = tidy[tidy["rel_path"].isin(needed_rel_paths)][["rel_path", "datetime", "DBT", "RH"]].copy()
    if tidy_sub.empty:
        return {"keys": [], "profiles": {}}

    tidy_sub = tidy_sub.merge(rel_map[["rel_path", "location_id", "role"]], on="rel_path", how="inner")
    tidy_sub["month"] = tidy_sub["datetime"].dt.month
    tidy_sub["hour"] = tidy_sub["datetime"].dt.hour

    agg = (
        tidy_sub.groupby(["location_id", "role", "month", "hour"], as_index=False)
        .agg(DBT=("DBT", "mean"), RH=("RH", "mean"))
    )

    keys = [(m, h) for m in range(1, 13) for h in range(0, 24)]
    key_index = {k: i for i, k in enumerate(keys)}

    name_by_loc = (
        idx_sub.groupby("location_id", as_index=False)["location_name"].first()
        .set_index("location_id")["location_name"]
        .to_dict()
    )

    profiles: Dict[str, Any] = {}
    for loc in locs:
        profiles[loc] = {
            "name": str(name_by_loc.get(loc) or loc),
            "base": {"DBT": [None] * 288, "RH": [None] * 288},
            "comp": {"DBT": [None] * 288, "RH": [None] * 288},
        }

    for _, r in agg.iterrows():
        loc = str(r["location_id"])
        role = str(r["role"])
        k = (int(r["month"]), int(r["hour"]))
        j = key_index.get(k)
        if j is None or loc not in profiles:
            continue
        profiles[loc][role]["DBT"][j] = None if pd.isna(r["DBT"]) else float(round(r["DBT"], 3))
        profiles[loc][role]["RH"][j] = None if pd.isna(r["RH"]) else float(round(r["RH"], 3))

    return {"keys": [[m, h] for (m, h) in keys], "profiles": profiles}


@st.cache_data(show_spinner=False)
def f120__build_daily_db_profiles(
    tidy: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_ids: Tuple[str, ...],
    baseline_variant: str,
    compare_variant: str,
    daily_stat: str,
) -> Dict[str, Any]:
    if daily_stat not in ("mean", "max"):
        raise ValueError("daily_stat must be 'mean' or 'max'")

    locs = [str(x) for x in location_ids]
    idx_sub = idx[idx["location_id"].isin(locs)].copy()
    if idx_sub.empty:
        return {"keys": [], "profiles": {}}

    base_rows = idx_sub[idx_sub["variant"] == baseline_variant].copy()
    if base_rows.empty:
        return {"keys": [], "profiles": {}}
    base_rows["base_group"] = base_rows["group_key"]
    base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
        base_location_id=("location_id", "first"),
        base_group=("group_key", "first"),
    )

    base_rels = base_rows[["location_id", "rel_path", "base_group"]].dropna().drop_duplicates()
    base_rels["role"] = "base"

    comp_rows = idx[idx["variant"] == compare_variant].copy()
    comp_rows = comp_rows.merge(base_group_by_key, on="group_key", how="inner")
    comp_rows = comp_rows[comp_rows["base_location_id"].isin(locs)]
    comp_rels = comp_rows[["rel_path"]].assign(
        location_id=comp_rows["base_location_id"], role="comp"
    ).dropna().drop_duplicates()

    rel_map = pd.concat([base_rels[["location_id", "rel_path", "role"]], comp_rels], ignore_index=True)
    if rel_map.empty:
        return {"keys": [], "profiles": {}}

    tidy_sub = tidy[tidy["rel_path"].isin(rel_map["rel_path"].unique().tolist())][
        ["rel_path", "datetime", "DBT"]
    ].copy()
    if tidy_sub.empty:
        return {"keys": [], "profiles": {}}

    tidy_sub = tidy_sub.merge(rel_map[["rel_path", "location_id", "role"]], on="rel_path", how="inner")
    tidy_sub["month"] = tidy_sub["datetime"].dt.month
    tidy_sub["day"] = tidy_sub["datetime"].dt.day
    tidy_sub["hour"] = tidy_sub["datetime"].dt.hour

    hourly = (
        tidy_sub.groupby(["location_id", "role", "month", "day", "hour"], as_index=False)
        .agg(DBT=("DBT", "mean"))
    )

    if daily_stat == "mean":
        daily = hourly.groupby(["location_id", "role", "month", "day"], as_index=False).agg(DBT=("DBT", "mean"))
    else:
        daily = hourly.groupby(["location_id", "role", "month", "day"], as_index=False).agg(DBT=("DBT", "max"))

    base_year = 2001
    all_days = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    keys = [(int(d.month), int(d.day)) for d in all_days]
    key_index = {k: i for i, k in enumerate(keys)}

    name_by_loc = (
        idx_sub.groupby("location_id", as_index=False)["location_name"].first()
        .set_index("location_id")["location_name"]
        .to_dict()
    )

    profiles: Dict[str, Any] = {}
    for loc in locs:
        profiles[loc] = {
            "name": str(name_by_loc.get(loc) or loc),
            "base": [None] * len(keys),
            "comp": [None] * len(keys),
        }

    for _, r in daily.iterrows():
        loc = str(r["location_id"])
        role = str(r["role"])
        k = (int(r["month"]), int(r["day"]))
        j = key_index.get(k)
        if j is None or loc not in profiles:
            continue
        v = None if pd.isna(r["DBT"]) else float(round(r["DBT"], 3))
        profiles[loc][role][j] = v

    return {"keys": [[m, d] for (m, d) in keys], "profiles": profiles}


@st.cache_data(show_spinner=False)
def f22c__build_daily_db_profiles_single_variant(
    tidy: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_ids: Tuple[str, ...],
    variant: str,
    daily_stat: str,
    baseline_variant: str | None = None,
) -> Dict[str, Any]:
    """
    Build per-location daily series for ONE variant (absolute values) to feed the D3 dashboard.

    Returns:
      {"keys": [[month, day], ...], "profiles": {loc_id: {"name": str, "series": [..]}}}
    """
    if daily_stat not in ("mean", "max"):
        raise ValueError("daily_stat must be 'mean' or 'max'")

    locs = [str(x) for x in location_ids]
    idx_sub = idx[idx["location_id"].isin(locs)].copy()
    if idx_sub.empty:
        return {"keys": [], "profiles": {}}

    if baseline_variant:
        base_rows = idx_sub[idx_sub["variant"] == baseline_variant].copy()
        if base_rows.empty:
            return {"keys": [], "profiles": {}}
        base_rows["base_group"] = base_rows["group_key"]
        base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
            base_location_id=("location_id", "first"),
        )
        rows = idx[idx["variant"] == variant].merge(base_group_by_key, on="group_key", how="inner")
        rows = rows[rows["base_location_id"].isin(locs)].copy()
        rows["location_id"] = rows["base_location_id"]
    else:
        rows = idx_sub[idx_sub["variant"] == variant].copy()
    if rows.empty:
        return {"keys": [], "profiles": {}}

    rels = rows[["location_id", "rel_path"]].dropna().drop_duplicates()
    if rels.empty:
        return {"keys": [], "profiles": {}}

    tidy_sub = tidy[tidy["rel_path"].isin(rels["rel_path"].unique().tolist())][
        ["rel_path", "datetime", "DBT"]
    ].copy()
    if tidy_sub.empty:
        return {"keys": [], "profiles": {}}

    tidy_sub = tidy_sub.merge(rels, on="rel_path", how="inner")
    tidy_sub["month"] = tidy_sub["datetime"].dt.month
    tidy_sub["day"] = tidy_sub["datetime"].dt.day
    tidy_sub["hour"] = tidy_sub["datetime"].dt.hour

    hourly = (
        tidy_sub.groupby(["location_id", "month", "day", "hour"], as_index=False)
        .agg(DBT=("DBT", "mean"))
    )

    if daily_stat == "mean":
        daily = hourly.groupby(["location_id", "month", "day"], as_index=False).agg(DBT=("DBT", "mean"))
    else:
        daily = hourly.groupby(["location_id", "month", "day"], as_index=False).agg(DBT=("DBT", "max"))

    base_year = 2001
    all_days = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    keys = [(int(d.month), int(d.day)) for d in all_days]
    key_index = {k: i for i, k in enumerate(keys)}

    name_by_loc = (
        idx_sub.groupby("location_id", as_index=False)["location_name"].first()
        .set_index("location_id")["location_name"]
        .to_dict()
    )

    profiles: Dict[str, Any] = {}
    for loc in locs:
        profiles[loc] = {
            "name": str(name_by_loc.get(loc) or loc),
            "series": [None] * len(keys),
        }

    for _, r in daily.iterrows():
        loc = str(r["location_id"])
        k = (int(r["month"]), int(r["day"]))
        j = key_index.get(k)
        if j is None or loc not in profiles:
            continue
        v = None if pd.isna(r["DBT"]) else float(round(r["DBT"], 3))
        profiles[loc]["series"][j] = v

    return {"keys": [[m, d] for (m, d) in keys], "profiles": profiles}


@st.cache_data(show_spinner=False)
def f22d__build_daily_db_profiles_from_daily_stats(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_ids: Tuple[str, ...],
    baseline_variant: str,
    compare_variant: str,
    daily_stat: str,
) -> Dict[str, Any]:
    """Daily profiles (baseline vs compare) built from precomputed daily stats."""
    if daily_stat not in ("mean", "max"):
        raise ValueError("daily_stat must be 'mean' or 'max'")
    if daily_stats is None or daily_stats.empty or idx is None or idx.empty:
        return {"keys": [], "profiles": {}}

    locs = [str(x) for x in location_ids]
    idx_sub = idx[idx["location_id"].isin(locs)].copy()
    if idx_sub.empty:
        return {"keys": [], "profiles": {}}

    base_rows = idx_sub[idx_sub["variant"] == baseline_variant].copy()
    if base_rows.empty:
        return {"keys": [], "profiles": {}}
    base_rows["base_group"] = base_rows["group_key"]
    base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
        base_location_id=("location_id", "first"),
        base_group=("group_key", "first"),
    )

    base_rels = base_rows[["location_id", "rel_path", "base_group"]].dropna().drop_duplicates()
    base_rels["role"] = "base"

    comp_rows = idx[idx["variant"] == compare_variant].copy()
    comp_rows = comp_rows.merge(base_group_by_key, on="group_key", how="inner")
    comp_rows = comp_rows[comp_rows["base_location_id"].isin(locs)]
    comp_rels = comp_rows[["rel_path"]].assign(
        location_id=comp_rows["base_location_id"], role="comp"
    ).dropna().drop_duplicates()

    rel_map = pd.concat([base_rels[["location_id", "rel_path", "role"]], comp_rels], ignore_index=True)
    if rel_map.empty:
        return {"keys": [], "profiles": {}}

    ds = daily_stats[daily_stats["rel_path"].isin(rel_map["rel_path"].unique().tolist())].copy()
    if ds.empty:
        return {"keys": [], "profiles": {}}
    ds = ds.merge(rel_map, on="rel_path", how="inner")

    col = "DBT_max" if daily_stat == "max" else "DBT_mean"
    daily = ds.groupby(["location_id", "role", "month", "day"], as_index=False).agg(DBT=(col, "mean"))

    base_year = 2001
    all_days = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    keys = [(int(d.month), int(d.day)) for d in all_days]
    key_index = {k: i for i, k in enumerate(keys)}

    name_by_loc = (
        idx_sub.groupby("location_id", as_index=False)["location_name"].first()
        .set_index("location_id")["location_name"]
        .to_dict()
    )

    profiles: Dict[str, Any] = {}
    for loc in locs:
        profiles[loc] = {
            "name": str(name_by_loc.get(loc) or loc),
            "base": [None] * len(keys),
            "comp": [None] * len(keys),
        }

    for _, r in daily.iterrows():
        loc = str(r["location_id"])
        role = str(r["role"])
        k = (int(r["month"]), int(r["day"]))
        j = key_index.get(k)
        if j is None or loc not in profiles:
            continue
        v = None if pd.isna(r["DBT"]) else float(round(r["DBT"], 3))
        profiles[loc][role][j] = v

    return {"keys": [[m, d] for (m, d) in keys], "profiles": profiles}


@st.cache_data(show_spinner=False)
def f22e__build_daily_db_profiles_single_variant_from_daily_stats(
    daily_stats: pd.DataFrame,
    idx: pd.DataFrame,
    *,
    location_ids: Tuple[str, ...],
    variant: str,
    daily_stat: str,
    baseline_variant: str | None = None,
) -> Dict[str, Any]:
    """Daily profiles (single variant) built from precomputed daily stats."""
    if daily_stat not in ("mean", "max"):
        raise ValueError("daily_stat must be 'mean' or 'max'")
    if daily_stats is None or daily_stats.empty or idx is None or idx.empty:
        return {"keys": [], "profiles": {}}

    locs = [str(x) for x in location_ids]
    idx_sub = idx[idx["location_id"].isin(locs)].copy()
    if idx_sub.empty:
        return {"keys": [], "profiles": {}}

    if baseline_variant:
        base_rows = idx_sub[idx_sub["variant"] == baseline_variant].copy()
        if base_rows.empty:
            return {"keys": [], "profiles": {}}
        base_rows["base_group"] = base_rows["group_key"]
        base_group_by_key = base_rows.groupby("group_key", as_index=False).agg(
            base_location_id=("location_id", "first"),
        )
        rows = idx[idx["variant"] == variant].merge(base_group_by_key, on="group_key", how="inner")
        rows = rows[rows["base_location_id"].isin(locs)].copy()
        rows["location_id"] = rows["base_location_id"]
    else:
        rows = idx_sub[idx_sub["variant"] == variant].copy()

    if rows.empty:
        return {"keys": [], "profiles": {}}

    rels = rows[["location_id", "rel_path"]].dropna().drop_duplicates()
    if rels.empty:
        return {"keys": [], "profiles": {}}

    ds = daily_stats[daily_stats["rel_path"].isin(rels["rel_path"].unique().tolist())].copy()
    if ds.empty:
        return {"keys": [], "profiles": {}}
    ds = ds.merge(rels, on="rel_path", how="inner")

    col = "DBT_max" if daily_stat == "max" else "DBT_mean"
    daily = ds.groupby(["location_id", "month", "day"], as_index=False).agg(DBT=(col, "mean"))

    base_year = 2001
    all_days = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    keys = [(int(d.month), int(d.day)) for d in all_days]
    key_index = {k: i for i, k in enumerate(keys)}

    name_by_loc = (
        idx_sub.groupby("location_id", as_index=False)["location_name"].first()
        .set_index("location_id")["location_name"]
        .to_dict()
    )

    profiles: Dict[str, Any] = {}
    for loc in locs:
        profiles[loc] = {
            "name": str(name_by_loc.get(loc) or loc),
            "series": [None] * len(keys),
        }

    for _, r in daily.iterrows():
        loc = str(r["location_id"])
        k = (int(r["month"]), int(r["day"]))
        j = key_index.get(k)
        if j is None or loc not in profiles:
            continue
        v = None if pd.isna(r["DBT"]) else float(round(r["DBT"], 3))
        profiles[loc]["series"][j] = v

    return {"keys": [[m, d] for (m, d) in keys], "profiles": profiles}


# -----------------------------
# Monthly metrics with threshold
# -----------------------------
def f126__calculate_monthly_metrics(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variants: list[str],
    threshold: float,
    metric_type: str = "above",  # "above" or "below"
) -> pd.DataFrame:
    """
    Calculate monthly metrics: hours and percentage above/below threshold for each variant.

    Returns a DataFrame with columns: variant, month, hours, percentage, total_hours
    """
    results = []

    for variant in variants:
        rel_paths = idx_df[
            (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
        ]["rel_path"].dropna().unique().tolist()
        if not rel_paths:
            continue

        ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
        if ts.empty:
            continue

        ts["month"] = ts["datetime"].dt.month
        ts = ts.dropna(subset=["DBT", "month"])

        # Filter by threshold
        if metric_type == "above":
            filtered = ts[ts["DBT"] >= threshold].copy()
        else:  # below
            filtered = ts[ts["DBT"] <= threshold].copy()

        # Calculate monthly totals
        monthly_counts = filtered.groupby("month", observed=True).size().reset_index(name="hours")
        monthly_totals = ts.groupby("month", observed=True).size().reset_index(name="total_hours")

        monthly = monthly_totals.merge(monthly_counts, on="month", how="left")
        monthly["hours"] = monthly["hours"].fillna(0).astype(int)
        monthly["percentage"] = (monthly["hours"] / monthly["total_hours"] * 100).round(2)
        monthly["variant"] = variant
        monthly["month_abbr"] = monthly["month"].apply(lambda x: calendar.month_abbr[int(x)])

        results.append(monthly[["variant", "month", "month_abbr", "hours", "percentage", "total_hours"]])

    if not results:
        return pd.DataFrame(columns=["variant", "month", "month_abbr", "hours", "percentage", "total_hours"])

    return pd.concat(results, ignore_index=True)


def f127__calculate_hours_above_thresholds(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variants: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Calculate total hours above each threshold for each variant.
    
    Returns a DataFrame with columns: variant, threshold, hours
    """
    results = []
    
    for variant in variants:
        rel_paths = idx_df[
            (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
        ]["rel_path"].dropna().unique().tolist()
        if not rel_paths:
            continue
        
        ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
        if ts.empty:
            continue
        
        ts = ts.dropna(subset=["DBT"])
        
        for threshold in thresholds:
            # Count hours above threshold
            hours_above = int((ts["DBT"] >= threshold).sum())
            results.append({
                "variant": variant,
                "threshold": threshold,
                "hours": hours_above,
            })
    
    if not results:
        return pd.DataFrame(columns=["variant", "threshold", "hours"])
    
    return pd.DataFrame(results)


def f002__custom_hr(margin_top: float = 0.5, margin_bottom: float = 0.5) -> None:
    """
    Render a custom horizontal line with configurable spacing.
    
    Args:
        margin_top: Top margin in rem units (default: 0.5)
        margin_bottom: Bottom margin in rem units (default: 0.5)
    """
    st.markdown(
        f"""
        <style>
            .custom-hr {{
                margin-top: {margin_top}rem;
                margin-bottom: {margin_bottom}rem;
                padding: 0;
                height: 1px;
                border: none;
                background-color: #ccc;
            }}
        </style>
        <div class="custom-hr"></div>
        """,
        unsafe_allow_html=True,
    )


def f003__vertical_spacing(height_px: float = 20.0) -> None:
    """
    Render a custom vertical spacing div for precise alignment.
    
    Args:
        height_px: Height in pixels (default: 20.0)
    """
    st.markdown(
        f"<div style='height: {height_px}px; width: 100%;'></div>",
        unsafe_allow_html=True,
    )

