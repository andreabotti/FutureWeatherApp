# Climate Data Visualization - Streamlit App

Streamlit application for visualizing and comparing Italian climate data from multiple sources: TMYx (Typical Meteorological Year), RCP climate scenarios (Representative Concentration Pathways), and CTI weather stations.

---

## 🚀 Quick Start

### One-Liner Commands

**TMYx↔FWG (B route, recommended):**
```bash
python data/data_preparation_scripts/05B_build_station_parquets.py --out data/04__italy_tmy_fwg_parquet && python data/data_preparation_scripts/06B_precompute_station_tables.py --data-dir data/04__italy_tmy_fwg_parquet && streamlit run app.py
```

**CTI (standalone EPW, B-style route):**
```bash
python data/data_preparation_scripts/05C_build_cti_station_parquets.py --cti-root data/01__italy_cti --out-root data/04__italy_cti_parquet && python data/data_preparation_scripts/06C_precompute_cti_tables.py --parquet-root data/04__italy_cti_parquet && streamlit run app.py
```

**Legacy EPW/TMYx (A route):**
```bash
python data/data_preparation_scripts/05A_build_epw_index_and_extract.py --root data/01__italy_epw_all --root data/02__italy_fwg_outputs --out data/03__italy_all_epw_DBT_streamlit --regional && python data/data_preparation_scripts/06A_precompute_derived_stats.py --data-dir data/03__italy_all_epw_DBT_streamlit --compute-aggregates && streamlit run app.py
```

**If Files Already Exist:**
```bash
streamlit run app.py
```

---

## 🧭 B Route Tables (Preferred)

Location: `data/04__italy_tmy_fwg_parquet/_tables/`

- `D-TMYxFWG__DBT__F-DD__L-ALL.parquet` — Daily stats (tidy): `region`, `station_key`, `scenario`, `date`, `Tmax`, `Tmean`, `Tmin`
- `D-TMYxFWG__Inventory__F-NA__L-ALL.parquet` — Station inventory + available scenario columns
- `pairing_debug.csv` — Missing/present scenario columns per station + baseline pairing

## 🧭 CTI Tables (Standalone)

Location: `data/04__italy_cti_parquet/_tables/`

- `D-CTI__DBT__F-DD__L-ALL.parquet` — Daily stats (tidy): `region`, `station_key`, `scenario`, `date`, `Tmax`, `Tmean`, `Tmin`
- `D-CTI__DBT__F-MM__L-ALL.parquet` — Monthly stats (tidy): `region`, `station_key`, `scenario`, `month`, `Tmax`, `Tmean`, `Tmin`
- `D-CTI__Inventory__F-NA__L-ALL.parquet` — Station inventory (`cols` is typically `cti`)

## 📁 Project Structure

```
FWG/
├── app.py                                  # Main Streamlit application
├── libs/
│   └── fn__libs.py                         # Helper functions library
├── data/
│   ├── 01__italy_epw_all/                  # Source EPW files (baseline climate)
│   ├── 02__italy_fwg_outputs/              # Future Weather Generator outputs (RCP scenarios)
│   ├── 04__italy_tmy_fwg_parquet/          # B route (per-station parquet + tables)
│   │   ├── <REGION>/<STATION_KEY>.parquet  # Hourly station parquet (wide columns)
│   │   └── _tables/                        # Precomputed tables for the app
│   ├── 01__italy_cti/                      # CTI weather station EPW files
│   │   ├── epw/                            # 110 weather station EPW files
│   │   └── CTI__list__ITA_WeatherStations__All.csv
│   ├── 04__italy_cti_parquet/              # CTI parquet output (per-station + tables)
│   │   ├── <REGION>/<STATION_KEY>.parquet  # Hourly station parquet (wide columns)
│   │   └── _tables/                        # Precomputed CTI tables
│   ├── 03__italy_all_epw_DBT_streamlit/    # Legacy app data directory (A route)
│   │   ├── epw_index.json                  # TMYx metadata
│   │   ├── cti_index.json                  # CTI metadata (if processed)
│   │   ├── DBT__HR__XX.parquet            # Regional hourly data (20 files)
│   │   ├── CTI__HR__XX.parquet            # CTI regional hourly (20 files, if processed)
│   │   ├── daily_stats.parquet            # Daily aggregates (REQUIRED)
│   │   └── *_by_*.parquet                 # Precomputed stats (optional)
│   └── data_preparation_scripts/           # Data processing scripts
│       ├── 05_build_epw_index_and_extract.py
│       ├── 05b_build_cti_epw_index_and_extract.py
│       └── 06_precompute_derived_stats.py
└── README.md                               # This file
```

---

## 🔧 Data Preparation Workflow

### B Route (TMYx↔FWG, preferred)

**Step 1: Build per-station parquets**
```bash
python data/data_preparation_scripts/05B_build_station_parquets.py --out data/04__italy_tmy_fwg_parquet
```

**Step 2: Precompute app tables**
```bash
python data/data_preparation_scripts/06B_precompute_station_tables.py --data-dir data/04__italy_tmy_fwg_parquet
```

**Output:**
- `data/04__italy_tmy_fwg_parquet/<REGION>/<STATION_KEY>.parquet`
- `data/04__italy_tmy_fwg_parquet/_tables/D-TMYxFWG__DBT__F-DD__L-ALL.parquet`
- `data/04__italy_tmy_fwg_parquet/_tables/D-TMYxFWG__Inventory__F-NA__L-ALL.parquet`
- `data/04__italy_tmy_fwg_parquet/_tables/pairing_debug.csv`

The Streamlit app prefers the `_tables` outputs for all summaries and only loads a per-station hourly parquet when a detailed station plot is requested.

---

### CTI Route (Standalone EPW)

**Step 1: Build per-station parquets**
```bash
python data/data_preparation_scripts/05C_build_cti_station_parquets.py \
  --cti-root data/01__italy_cti \
  --out-root data/04__italy_cti_parquet
```

**Step 2: Precompute app tables**
```bash
python data/data_preparation_scripts/06C_precompute_cti_tables.py \
  --parquet-root data/04__italy_cti_parquet
```

**Output:**
- `data/04__italy_cti_parquet/<REGION>/<STATION_KEY>.parquet`
- `data/04__italy_cti_parquet/_tables/D-CTI__DBT__F-DD__L-ALL.parquet`
- `data/04__italy_cti_parquet/_tables/D-CTI__DBT__F-MM__L-ALL.parquet`
- `data/04__italy_cti_parquet/_tables/D-CTI__Inventory__F-NA__L-ALL.parquet`

The app never computes CTI daily/monthly stats at runtime; it always uses the precomputed `_tables`.

---

### Step 1: Extract Hourly Data (REQUIRED)

**For TMYx/EPW data:**
```bash
python data/data_preparation_scripts/05_build_epw_index_and_extract.py \
  --root data/01__italy_epw_all \
  --root data/02__italy_fwg_outputs \
  --out data/03__italy_all_epw_DBT_streamlit \
  --regional
```

**Creates:**
- `epw_index.json` - Metadata for ~4,144 EPW files
- `DBT__HR__AB.parquet` through `DBT__HR__VN.parquet` - 20 regional files (97 MB total)

**Time:** ~10-60 minutes (depending on EPW file count)

---

### Step 2: Process CTI Data (OPTIONAL)

**First, rename CTI files (one-time setup):**
```bash
cd data/01__italy_cti/epw
python rename_cti_regions.py
cd ../../..
```

**Then extract CTI hourly data:**
```bash
python data/data_preparation_scripts/05b_build_cti_epw_index_and_extract.py \
  --root data/01__italy_cti/epw \
  --out data/03__italy_all_epw_DBT_streamlit \
  --regional
```

**Creates:**
- `cti_index.json` - Metadata for 110 weather stations
- `CTI__HR__AB.parquet` through `CTI__HR__VN.parquet` - 20 regional CTI files

**Time:** ~2-5 minutes

---

### Step 3: Precompute Derived Statistics (STRONGLY RECOMMENDED)

**Basic (daily stats only):**
```bash
python data/data_preparation_scripts/06_precompute_derived_stats.py \
  --data-dir data/03__italy_all_epw_DBT_streamlit
```

**Recommended (with aggregates for maximum performance):**
```bash
python data/data_preparation_scripts/06_precompute_derived_stats.py \
  --data-dir data/03__italy_all_epw_DBT_streamlit \
  --compute-aggregates
```

**With CTI data:**
```bash
python data/data_preparation_scripts/06_precompute_derived_stats.py \
  --data-dir data/03__italy_all_epw_DBT_streamlit \
  --process-cti \
  --compute-aggregates
```

**Creates:**
- `daily_stats.parquet` - Daily aggregates with scenario column (REQUIRED - eliminates 27s+ startup)
- `file_stats_by_percentile.parquet` - File-level statistics
- `location_stats_by_variant_percentile.parquet` - Location statistics
- `location_deltas_by_variant_pair_percentile.parquet` - Variant comparisons
- `monthly_delta_tables_by_variant_pair_percentile_metric.parquet` - Monthly tables

**Time:** 
- Basic: ~2-5 minutes
- With aggregates: ~7-20 minutes (but makes app near-instant)

---

### Step 4: Launch the App

```bash
streamlit run app.py
```

**Expected startup time:**
- Without precomputation: 45-60 seconds ❌
- With daily_stats.parquet: 5-10 seconds ✅
- With all aggregates: 2-5 seconds ⚡

---

## 📊 Scripts Consolidation

All data preprocessing is now handled by **one unified script**: `06_precompute_derived_stats.py`

### What Happened to 00_ and 01_ Scripts?

**Deprecated:**
- ❌ `00_precompute_daily_stats.py` - Superseded by 06_
- ❌ `01_precompute_aggregated_stats.py` - Merged into 06_

**Current:**
- ✅ `06_precompute_derived_stats.py` - Handles ALL derived statistics

### New Unified Workflow

```
Hourly Data (DBT__HR__*.parquet, CTI__HR__*.parquet)
    ↓
06_precompute_derived_stats.py
    ├─ Compute Daily Stats (always)
    ├─ Process CTI data (if --process-cti)
    └─ Compute Aggregates (if --compute-aggregates)
        ↓
        ├─ File stats by percentile
        ├─ Location stats by variant & percentile
        ├─ Location deltas by variant pair & percentile
        └─ Monthly delta tables
    ↓
Streamlit App (fast/near-instant startup)
```

### Benefits

✅ **Single logical flow** - All derived statistics in one place  
✅ **Simpler workflow** - No need to remember multiple scripts  
✅ **Better documentation** - Clear what each flag does  
✅ **Easier maintenance** - Changes in one file  
✅ **User-friendly** - One command with optional flags

---

## 🗂️ Regional vs Per-Station Format

### Why Regional Format?

After testing both approaches, **regional format is 5.7x more efficient**:

| Format | Files | Total Size | Notes |
|--------|-------|------------|-------|
| **Regional** ✅ | 20 | 97 MB | RECOMMENDED |
| Per-Station ❌ | 4,144 | 554 MB | NOT recommended |

### Why Such a Difference?

1. **Parquet Compression:** Columnar storage with dictionary encoding compresses repeated values (`rel_path`, `scenario`) to nearly zero bytes
2. **File System Overhead:** 4,144 files = significant metadata overhead vs 20 files
3. **Parquet File Overhead:** Each file has header/footer, schema, statistics - multiplied 207x with per-station approach

### Solution

Keep the efficient regional format + enhance Streamlit app with:
- Location filter (select specific stations)
- Scenario filter (TMYx, RCP, CTI)
- Real-time statistics

✅ **Best of both worlds:** Optimal storage + flexible data exploration

---

## 🌡️ CTI Weather Station Data

### Overview

CTI (Comitato Termotecnico Italiano) weather station data provides real observed climate data for 110 Italian locations.

### Processing Steps

#### 1. Rename Files (One-Time Setup)

CTI files use 3-letter codes (ABR, BAS, CAL), but the app uses 2-letter codes (AB, BC, LB).

```bash
cd data/01__italy_cti/epw
python rename_cti_regions.py
```

**Mapping Examples:**
- `ABR` → `AB` (Abruzzo)
- `BAS` → `BC` (Basilicata)
- `CAL` → `LB` (Calabria)
- `LAZ` → `LZ` (Lazio)
- `SIC` → `SC` (Sicilia)

*(See full mapping table in the script)*

#### 2. Extract CTI Data

```bash
python data/data_preparation_scripts/05b_build_cti_epw_index_and_extract.py \
  --root data/01__italy_cti/epw \
  --out data/03__italy_all_epw_DBT_streamlit \
  --regional
```

**Output:**
- `cti_index.json` - Station metadata (lat/lon/alt)
- `CTI__HR__XX.parquet` - Regional hourly data (20 files)

#### 3. Integrate with TMYx

Run `06_` with `--process-cti` flag to merge CTI and TMYx data:

```bash
python data/data_preparation_scripts/06_precompute_derived_stats.py \
  --data-dir data/03__italy_all_epw_DBT_streamlit \
  --process-cti \
  --compute-aggregates
```

### CTI Data Format

**CTI Regional Files (`CTI__HR__XX.parquet`):**
- **Index:** `datetime` (hourly, 8760 per year)
- **Columns:** `DBT`, `location_id`

**CTI Index (`cti_index.json`):**
```json
{
  "location_id": "AB__AQ__L'Aquila",
  "location_name": "L'Aquila",
  "region": "AB",
  "latitude": 42.1368853,
  "longitude": 13.6103410,
  "altitude": 700.0,
  "source": "CTI"
}
```

### Benefits

1. **Real Observed Data:** Actual measurements from Italian weather stations
2. **Consistent Format:** Matches TMYx data structure
3. **Regional Organization:** Easy to load specific regions
4. **App Integration:** Automatically detected in Data Preview tab

---

## ⚡ Performance Comparison

| Workflow | App Startup | Widget Changes | Recommended |
|----------|-------------|----------------|-------------|
| **No precomputation** | 45-60s | 15-30s | ❌ |
| **With daily_stats** | 5-10s | 5-10s | ⚠️ Minimum |
| **With all aggregates** | 2-5s | 0.1-0.5s | ✅ Best |

### What Gets Precomputed?

**Daily stats (`daily_stats.parquet`):**
- Eliminates 27+ seconds of app startup
- Required for variant filtering

**Aggregates (with `--compute-aggregates`):**
- File stats by percentile (95%, 97.5%, 99%)
- Location stats for all variants
- Location deltas for all variant pairs
- Monthly delta tables for all combinations

**Impact:**
- 5-10x faster widget interactions
- Near-instant responses for cached computations
- Makes the app feel responsive and professional

---

## 🔍 Data Explorer

The Streamlit app includes a "Data Preview" tab with:

### Regional Data Explorer
- **Subtabs per Italian region** (Abruzzo, Basilicata, etc.) + CTI tab
- **Location selector** - Choose specific weather station
- **Scenario plots** - Automatic plots for each available scenario (TMYx, RCP45, RCP85, CTI)
- **Statistics panel** - DBT stats, time range, data quality

### Benefits
- Troubleshoot data issues by location
- Compare scenarios visually
- Verify data completeness
- No need to export or use external tools

---

## 🛠️ Troubleshooting

### App Issues

**"daily_stats.parquet not found"**
```bash
python data/data_preparation_scripts/06_precompute_derived_stats.py --data-dir data/03__italy_all_epw_DBT_streamlit
```

**"No matching locations found between baseline and compare variant"**
- Ensure `daily_stats.parquet` has a `scenario` column
- Re-run `06_` to regenerate with proper schema

**Slow app startup (20+ seconds)**
- Run `06_` with `--compute-aggregates` flag for maximum performance

### Data Preparation Issues

**"No EPW files found"**
- Check that EPW files exist in `data/01__italy_epw_all/` or `data/02__italy_fwg_outputs/`
- Verify `--root` paths are correct (relative to FWG root directory)

**"Missing root(s)" error in 05_ script**
- Run script from FWG root directory, not from `data_preparation_scripts/`
- OR adjust `--root` paths to be relative to your current directory

**CTI files not renamed**
```bash
cd data/01__italy_cti/epw
python rename_cti_regions.py --dry-run  # Preview changes
python rename_cti_regions.py             # Apply changes
```

**CTI processing fails**
- Ensure `CTI__list__ITA_WeatherStations__All.csv` exists in `data/01__italy_cti/`
- Check that EPW files are named like `AB__AQ__Station.epw` (2-letter region code)

### Performance Issues

**Still slow after precomputation**
- Clear Streamlit cache: Check sidebar for cache controls
- Verify all `*_by_*.parquet` files exist in data directory
- Check file sizes: Regional files should be ~5-10 MB each

**Large file sizes**
- Regional format (97 MB for 20 files) is optimal
- Do NOT use `--convert-to-per-station` (creates 554 MB for 4,144 files)

---

## 📚 Additional Documentation

### Script-Specific Help
```bash
python data/data_preparation_scripts/05_build_epw_index_and_extract.py --help
python data/data_preparation_scripts/05b_build_cti_epw_index_and_extract.py --help
python data/data_preparation_scripts/06_precompute_derived_stats.py --help
```

### In-App Documentation
- **"Preparation Scripts" tab** - Complete workflow documentation
- **"Code Performance" tab** - See timing breakdowns
- **"Data Preview" tab** - Explore data by region/location
- **"Debug Info" tab** - Index contents, file info

---

## 🎯 Recommended Workflow Summary

```bash
# 1. Extract TMYx hourly data (REQUIRED)
python data/data_preparation_scripts/05_build_epw_index_and_extract.py \
  --root data/01__italy_epw_all \
  --root data/02__italy_fwg_outputs \
  --out data/03__italy_all_epw_DBT_streamlit \
  --regional

# 2. Extract CTI data (OPTIONAL)
cd data/01__italy_cti/epw && python rename_cti_regions.py && cd ../../..
python data/data_preparation_scripts/05b_build_cti_epw_index_and_extract.py \
  --root data/01__italy_cti/epw \
  --out data/03__italy_all_epw_DBT_streamlit \
  --regional

# 3. Precompute all statistics (RECOMMENDED)
python data/data_preparation_scripts/06_precompute_derived_stats.py \
  --data-dir data/03__italy_all_epw_DBT_streamlit \
  --process-cti \
  --compute-aggregates

# 4. Launch app
streamlit run app.py
```

**Total time:** ~20-40 minutes (one-time setup)  
**App performance:** Near-instant (2-5s startup, <0.5s interactions)

---

## 📝 Technical Details

### Data Format
- **Hourly data:** Parquet files with datetime index, columnar storage
- **Regional partitioning:** 20 files (one per Italian region)
- **Compression:** Dictionary encoding for repeated values
- **Columns:** `DBT` (temperature), `rel_path` (file ID), `scenario` (TMYx/RCP/CTI)

### Italian Region Codes
AB, BC, CM, ER, FV, LB, LG, LM, LZ, MH, ML, PM, PU, SC, SD, TC, TT, UM, VD, VN

### Data Sources
- **TMYx:** Typical Meteorological Year datasets (baseline climate)
- **RCP:** Representative Concentration Pathways (future climate scenarios)
  - RCP4.5 (2030, 2050, 2070)
  - RCP8.5 (2030, 2050, 2080)
- **CTI:** Comitato Termotecnico Italiano weather stations (110 locations)

---

## 🤝 Contributing

When modifying the data pipeline:
1. Test with a small subset first (`--limit 10` flag in 05_ scripts)
2. Verify `daily_stats.parquet` includes `scenario` column
3. Run with `--verbose` to see detailed processing info
4. Check app startup time to confirm performance improvements

---

## 📄 License

Project developed for EETRA srl SB - Climate Data Analysis and Visualization
