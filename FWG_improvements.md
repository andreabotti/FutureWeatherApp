# FWG Streamlit App — Speed & UI Improvement Analysis

> Code review of `app.py` (2 444 lines), `libs/fn__libs.py` (3 543 lines),  
> `libs/fn__libs_charts.py` (4 699 lines). Parquet data structure inferred from file names.

---

## 1. Critical Bugs Found

### 1.1 `r=200` in region map SVG circles (fn__libs_charts.py ~f206)

```python
# CURRENT – circles are 200 SVG units wide → fills entire region map
.attr("r", 200)
```
Every station marker in the zoomed-region map has radius 200px, making the circles fill the entire map viewport. Should be `4.6` (same as the national map).

### 1.2 Duplicate `with charts_col:` block in `_render_future_scenario_tab()`

`app.py` lines ~450–560 contain the entire Altair charts block **twice** inside the same `if render_mode.startswith("Plotly"):` branch, with the second block unreachable dead code. Remove the second `with charts_col:` block.

### 1.3 `fn__libs_charts.py` functions are never called by `app.py`

`fn__libs_charts.py` defines **f201–f213** (plotly maps, D3 dashboards, heatmap/scatter/stacked subplots). `app.py` calls the **old** `fn__libs.py` versions (`h.f19`, `h.f23`, `h.f34`, `h.f35`, `h.f36`, etc.). The refactored chart module is dead weight loaded on every import. Either complete the migration or delete the file.

---

## 2. Performance Issues

### 2.1 `iterrows()` in hourly series builder — **most critical hotspot**

```python
# _build_hourly_series_by_location_id, app.py
series = [
    {"t": row["datetime"].isoformat() ..., "v": float(row["DBT"])}
    for _, row in sub.iterrows()          # ← ~8 760 iterations per scenario
    if pd.notna(row.get("DBT"))
]
```

`iterrows()` is 10–100× slower than vectorized Pandas. With summer data (~2 200 rows) and 3 scenarios for 12 stations, this runs ~80 000 iterations on every first render.

**Fix — vectorized:**

```python
def _rows_to_series(df: pd.DataFrame) -> list[dict]:
    mask = df["DBT"].notna()
    df = df[mask]
    ts = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").values
    vs = df["DBT"].values.round(2)
    return [{"t": t, "v": float(v)} for t, v in zip(ts, vs)]
```

Or better, pass typed arrays to D3 instead of a list of `{t, v}` dicts:

```python
# Compact payload: two parallel arrays → 40% smaller JSON, faster parse
{"ts": ts_list, "vs": vs_list}
```

### 2.2 `importlib.reload()` on every Streamlit rerun

```python
# app.py top level — runs on EVERY browser interaction
import libs.fn__libs_charts as _fn_charts
importlib.reload(_fn_charts)
h = importlib.reload(h)
```

These two `reload()` calls re-parse 8 000+ lines of Python on every rerun (button click, sidebar widget, etc.). This is a development convenience that should be behind an env flag.

**Fix:**
```python
import os
if os.environ.get("FWG_DEV_RELOAD"):
    importlib.reload(_fn_charts)
    h = importlib.reload(h)
```

### 2.3 D3 GeoJSON fetched on every component render

Every embedded D3 HTML fetches two heavy GeoJSON files from CDN **inside the iframe**:
- `world-atlas@2/countries-50m.json` (~400 KB)
- `openpolis/geojson-italy/limits_IT_regions.geojson` (~2 MB)
- `openpolis/geojson-italy/limits_IT_provinces.geojson` (~4 MB)

These are re-fetched every time the user changes a sidebar option, because `components.html()` creates a new iframe. With 3 embedded dashboards open simultaneously, that's 6 × (400 KB + 2 MB + 4 MB) = ~40 MB of repeated downloads per page load.

**Fix — inline GeoJSON at startup:**

```python
# In fn__libs.py, cache at startup
@st.cache_data(show_spinner=False)
def _load_geojson_bundle() -> dict:
    """Load and simplify GeoJSON once at startup."""
    import requests, json, topojson as tp  # pip install topojson
    regions = requests.get(REGIONS_URL, timeout=10).json()
    # Simplify geometry 50% to reduce payload
    simplified = tp.Topology(regions, prequantize=False).toposimplify(0.01).to_geojson()
    return {"regions": json.loads(simplified)}

# Pass as JSON arg to the HTML template — eliminates network fetch inside iframe
geojson_bundle = _load_geojson_bundle()
```

Or use the simpler approach: **serve a static GeoJSON file from Streamlit** using `st.static_file_serving` (Streamlit ≥ 1.31) and reference it with a relative URL so it's browser-cached.

### 2.4 `_build_region_maps_html` cache key includes full JSON strings

```python
@st.cache_data(show_spinner=False)
def _build_region_maps_html(
    ...
    abs_points_json: str,      # can be megabytes
    profiles_json: str,        # can be megabytes
    ...
) -> str:
```

Streamlit's `@st.cache_data` hashes all arguments to build the cache key. Hashing multi-megabyte JSON strings on every rerun is slow even when the cache hits. Pass stable scalar keys instead:

```python
# Better: hash only a fingerprint, keep data in session_state
import hashlib

def _json_fingerprint(data: dict) -> str:
    raw = json.dumps(data, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()

@st.cache_data(show_spinner=False)
def _build_region_maps_html(
    fingerprint: str,          # stable hash, fast to hash
    ...                        # scalar params only
) -> str:
```

### 2.5 Profile data volume — 365 × N_locations × N_variants

The `profiles_bundle_by_variant` dict serialized to JSON contains daily profile series for **all** locations in all three scenario variants. If there are 50 locations, that's 50 × 365 × 3 ≈ 54 750 numbers per metric, serialized to a string. The entire payload is embedded in HTML and parsed by the browser on every component mount.

**Improvements:**
- Only include locations in the **currently selected region** (filter server-side before serialization)
- Use `Float32Array` encoding (base64) instead of JSON numbers — ~4× smaller
- Stream additional locations on demand via `postMessage` from Python

### 2.6 Large JSON being passed through `st.session_state` on click callbacks

The D3 component uses `postMessage` to communicate clicks back to Python, which triggers a full Streamlit rerun. All heavy data is re-serialized on every rerun. Consider using `st.query_params` for location selection (survives reload, avoids rerun).

---

## 3. D3 / JavaScript Improvements

### 3.1 Replace multiple `components.html()` embeds with a single custom component

Currently up to **3 separate D3 iframes** can be active simultaneously (Future Temperatures tab, Future vs Current tab, Italian Regions tab). Each has its own copy of D3, TopoJSON, and GeoJSON. This is:
- ~1.5 MB of duplicate JS per extra tab visit
- No shared state between maps
- Click callbacks require full page reruns

**Recommended architecture:**

```
streamlit_fwg_component/
├── __init__.py          # st.components.v1.declare_component
├── frontend/
│   ├── index.html
│   ├── main.js          # Single D3 app, receives all data via props
│   └── italy.topo.json  # Pre-simplified TopoJSON bundled at build time
```

A single `declare_component` with bidirectional messaging:
- Python sends scenario/region selection → component re-renders in place
- Component sends clicked `location_id` back → Python updates charts without full rerun

### 3.2 Use D3 `zoom()` for pan/zoom on the national map

The current national map is static SVG with no zoom. Given Italy's elongated shape (Sardinia/Sicily vs Alps), markers in dense areas (Po Valley) overlap heavily. Adding `d3.zoom()` costs ~20 lines of JS.

### 3.3 Improve map tooltip positioning

Current tooltip uses `event.pageX` + `event.pageY` which gives wrong positions inside Streamlit's iframe. Replace with:

```javascript
// Use bounding rect of the SVG + event offset
const svgRect = svg.node().getBoundingClientRect();
tip
  .style("left", (svgRect.left + event.offsetX + 12) + "px")
  .style("top", (svgRect.top + event.offsetY + 12) + "px");
```

### 3.4 Virtualize the label collision detection

The current label collision detection in `f207` iterates all `O(n²)` label pairs on every `renderRegion()` call. For 20+ stations in a region this is noticeable. Use a spatial index (simple grid bucket) or D3's `forceSimulation` for label placement.

### 3.5 Precompute month-level aggregates in Python, not JS

The D3 code recomputes `monthStat()` and `monthDelta()` in JavaScript from raw daily series on every click. With 365 data points per location/variant, this is fast, but the code is complex and duplicated. Instead, precompute monthly summaries in Python (already done for some paths) and pass only 12 monthly values per location/scenario. This reduces JS complexity and JSON payload.

### 3.6 Replace external icon URLs with inline SVG in chart titles

```javascript
// CURRENT — fetches external JPEGs as station/scenario icons (slow, may fail)
const stationIcon = `<img src="https://external-content.duckduckgo.com/...">`;
const scenarioIcon = `<img src="https://as2.ftcdn.net/...">`;
```

Replace with inline SVG icons (0 network requests, no external dependency):

```javascript
const stationIcon = `<svg width="14" height="14" viewBox="0 0 24 24" ...>...</svg>`;
```

---

## 4. UI / UX Improvements

### 4.1 Remove the "Show Charts" toggle friction

Every interactive tab hides its content behind a toggle:
```python
show_d3_regions_v2 = st.toggle("Show Charts", value=False, ...)
if not show_d3_regions_v2:
    st.info("Enable 'Show Charts' toggle to load interactive JavaScript charts.")
```
This was presumably added to avoid loading heavy data on inactive tabs. The better pattern is **tab-based lazy loading** — only load data when the user actually switches to that tab. Streamlit's `st.tabs()` already does this if you structure the data loading inside the `with tab:` block.

**Fix:**
```python
tab1, tab2, tab3 = st.tabs(["Future Temps", "Delta", "Italian Regions"])
with tab1:
    # Data only loaded when tab1 is active
    _render_future_scenario_tab()
```
This eliminates all toggles while keeping performance.

### 4.2 Sidebar: consolidate RCP + Year into a single radio

The RCP and Year radios in two columns create 4 widgets for what is effectively a single parameter (scenario). Consider a single radio with formatted labels:

```python
scenario_options = {
    "tmyx__rcp26_2050": "RCP 2.6 — 2050 (near future, low emissions)",
    "tmyx__rcp45_2050": "RCP 4.5 — 2050 (medium term)",
    "tmyx__rcp85_2080": "RCP 8.5 — 2080 (long term, high emissions)",
}
compare_variant = st.radio("Climate Scenario", list(scenario_options.keys()),
                           format_func=lambda k: scenario_options[k])
```

### 4.3 D3 region selector sync via `localStorage` is fragile in Streamlit

The region selector syncs between iframes using `localStorage` events. This is unreliable in Streamlit Cloud (sandboxed iframes, storage partitioned by origin). Use `st.query_params` instead:

```python
# Python side
if "region" not in st.query_params:
    st.query_params["region"] = "LM"
current_region = st.query_params["region"]

# D3 side: read from URL param, write back on selection change
const urlParams = new URLSearchParams(window.location.search);
const regionFromUrl = urlParams.get("region");
```

### 4.4 TMYx tab: extract inline CSS `<style>` blocks to page-level injection

The TMYx tab injects a `<style>` block via `st.markdown(..., unsafe_allow_html=True)` to override `.stMetric` styles. This causes visual flicker because it applies after the metrics render. Inject these styles once via `f101__inject_inter_font()` or a dedicated page-level CSS injection at startup.

### 4.5 Add a loading skeleton for D3 iframes

Currently, when D3 charts load (GeoJSON fetch can take 1–3 s), the component area is blank. Add a lightweight placeholder:

```python
placeholder = st.empty()
placeholder.info("🗺 Loading map data…")
html = _build_region_maps_html(...)
placeholder.empty()
st.components.v1.html(html, height=1150)
```

### 4.6 Table views: add column sorting and filtering

The `st.dataframe()` tables (delta table, monthly table) are read-only. Use `st.data_editor` with `disabled=True` or `AgGrid` for sortable, filterable tables that let users explore the data.

---

## 5. Architecture / Code Quality

### 5.1 Two parallel chart libraries — pick one

| File | Functions | Used by app.py? |
|------|-----------|-----------------|
| `fn__libs.py` | `f19`, `f23`, `f23b`, `f23d`, `f34`–`f36` | ✅ Yes |
| `fn__libs_charts.py` | `f201`–`f213` | ❌ No |

`fn__libs_charts.py` appears to be a clean-room rewrite of the chart functions (possibly started as a refactor), but `app.py` still calls the old functions. Either:
- **Migrate** `app.py` to use `f2xx` functions (they are better structured)
- Or **delete** `fn__libs_charts.py` to avoid maintenance confusion

### 5.2 Precompute and bundle GeoJSON at data preparation time

Add a step `06E_precompute_geojson.py` that:
1. Downloads Italy regions + provinces GeoJSON
2. Simplifies geometry with `topojson` (reduces size 60–80%)
3. Saves to `data/04__italy_tmy_fwg_parquet/italy_geo.json`

Then embed this file's contents directly in the HTML template (one-time at startup, already `@st.cache_data`-able), eliminating all CDN fetches from D3 code.

### 5.3 Type annotations and helper extraction

`_build_hourly_series_by_location_id` is 80+ lines and combines file discovery, hourly loading, summer filtering, JSON serialization, and alias mapping. Split into focused helpers:

```python
def _load_station_summer_hourly(path, station_key, scenarios) -> pd.DataFrame: ...
def _df_to_hourly_series(df: pd.DataFrame) -> dict[str, list]: ...
def _add_scenario_aliases(series: dict) -> dict: ...
```

### 5.4 Replace `time.perf_counter` timing with a context manager

Current timing pattern:
```python
start = time.perf_counter()
result = fn()
_record_timing(name, time.perf_counter() - start, notes=notes)
```

Replace with a cleaner context manager:
```python
@contextmanager
def _timer(name: str, notes: str = ""):
    t0 = time.perf_counter()
    yield
    _record_timing(name, time.perf_counter() - t0, notes)

with _timer("load_b_inventory"):
    inventory = _load_inventory_cached(DEFAULT_B_DATA_DIR)
```

---

## 6. Prioritized Action Plan

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 🔴 Critical | Fix `r=200` circle bug | 1 line | Map usable |
| 🔴 Critical | Remove `importlib.reload()` from hot path | 5 min | Rerun time −20% |
| 🔴 Critical | Replace `iterrows()` with vectorized series builder | 1 hour | −80% build time |
| 🟠 High | Inline/precompute GeoJSON to eliminate CDN fetches | 2 hours | −3 s load time |
| 🟠 High | Remove duplicate `with charts_col:` block | 5 min | Code clarity |
| 🟠 High | Remove toggles, use tab-level lazy loading | 2 hours | Better UX |
| 🟠 High | Hash fingerprint instead of full JSON in cache key | 1 hour | Cache hit speed |
| 🟡 Medium | Migrate `app.py` to `fn__libs_charts.py` functions or delete | 1 day | Maintainability |
| 🟡 Medium | D3 single component with `declare_component` | 3 days | −60% JS payload |
| 🟡 Medium | Add `d3.zoom()` to national map | 3 hours | Dense area exploration |
| 🟡 Medium | Fix tooltip `pageX`/`pageY` iframe bug | 30 min | Tooltip accuracy |
| 🟡 Medium | Replace external icon `<img>` with inline SVG | 30 min | Reliability |
| 🟢 Low | Add loading skeleton | 1 hour | UX polish |
| 🟢 Low | Add `st.data_editor` sortable tables | 2 hours | Data exploration |
| 🟢 Low | Extract `_timer` context manager | 30 min | Code quality |

---

## 7. Quick-Win Code Snippets

### 7.1 Vectorized summer series builder (replace iterrows)

```python
def _build_summer_series(df: pd.DataFrame) -> dict[str, list]:
    """Fast vectorized version of the inner series-building loop."""
    df = df.copy()
    df["_month"] = pd.to_datetime(df["datetime"], errors="coerce").dt.month
    summer = df[df["_month"].isin([6, 7, 8]) & df["DBT"].notna()]
    
    result = {}
    for sc, grp in summer.groupby("scenario", observed=True):
        sc_str = str(sc).strip()
        if not sc_str:
            continue
        ts = grp["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
        vs = grp["DBT"].round(2).tolist()
        # Compact format: 2 parallel arrays instead of N dicts
        result[sc_str] = {"ts": ts, "vs": vs}
    return result
```

Update D3 to consume `{"ts": [...], "vs": [...]}` instead of `[{"t": ..., "v": ...}, ...]`.

### 7.2 Disable hot-reload in production

```python
# app.py top — replace unconditional reloads
import os
_DEV = os.environ.get("FWG_DEV_RELOAD", "").lower() in ("1", "true", "yes")
if _DEV:
    import importlib as _il
    _fn_charts = _il.reload(_fn_charts)
    h = _il.reload(h)
```

### 7.3 Precompute GeoJSON once at startup

```python
# fn__libs.py
@st.cache_resource   # shared across all sessions, loaded once per worker
def _italy_geojson_bundle() -> dict:
    """Load and simplify Italy GeoJSON files once at startup."""
    import json, urllib.request
    def _fetch(url):
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    try:
        regions = _fetch(
            "https://raw.githubusercontent.com/openpolis/geojson-italy/"
            "master/geojson/limits_IT_regions.geojson"
        )
        provinces = _fetch(
            "https://raw.githubusercontent.com/openpolis/geojson-italy/"
            "master/geojson/limits_IT_provinces.geojson"
        )
    except Exception:
        return {}
    return {"regions": regions, "provinces": provinces}

# Pass as pre-serialized JSON arg to HTML template
# → eliminates async fetch inside iframe on every render
```

---

*Analysis based on static code review of 10 686 lines across app.py, fn__libs.py, fn__libs_charts.py, and the data directory structure. Runtime profiling with actual data is recommended to confirm the relative timings of each bottleneck.*
