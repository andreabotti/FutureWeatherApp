# Data Preparation Scripts

> **📖 For complete documentation, see the main [README.md](../../README.md) in the project root.**

This directory contains Python scripts for preparing climate data before running the Streamlit app.

## Quick Reference

### Required Scripts (Run in Order)

**1. Extract TMYx Hourly Data:**
```bash
python 05_build_epw_index_and_extract.py --root data/01__italy_epw_all --root data/02__italy_fwg_outputs --out data/03__italy_all_epw_DBT_streamlit --regional
```

**2. Extract CTI Data (Optional):**
```bash
cd data/01__italy_cti/epw && python rename_cti_regions.py && cd ../../..
python 05b_build_cti_epw_index_and_extract.py --root data/01__italy_cti/epw --out data/03__italy_all_epw_DBT_streamlit --regional
```

**3. Precompute Statistics (Strongly Recommended):**
```bash
python 06_precompute_derived_stats.py --data-dir data/03__italy_all_epw_DBT_streamlit --process-cti --compute-aggregates
```

## Script Overview

| Script | Purpose | Required? |
|--------|---------|-----------|
| `05_build_epw_index_and_extract.py` | Extract TMYx hourly data to regional parquet files | **Yes** |
| `05b_build_cti_epw_index_and_extract.py` | Extract CTI weather station data | Optional |
| `06_precompute_derived_stats.py` | Compute daily stats + aggregates | **Strongly Recommended** |

### Deprecated Scripts

The following scripts have been consolidated into `06_precompute_derived_stats.py`:
- ~~`00_precompute_daily_stats.py`~~ → Use `06_` instead
- ~~`01_precompute_aggregated_stats.py`~~ → Use `06_ --compute-aggregates` instead

## Quick Help

**Get script help:**
```bash
python <script_name>.py --help
```

**One-liner (TMYx only):**
```bash
python 05_build_epw_index_and_extract.py --root data/01__italy_epw_all --root data/02__italy_fwg_outputs --out data/03__italy_all_epw_DBT_streamlit --regional && python 06_precompute_derived_stats.py --data-dir data/03__italy_all_epw_DBT_streamlit --compute-aggregates
```

**One-liner (TMYx + CTI):**
```bash
python 05_build_epw_index_and_extract.py --root data/01__italy_epw_all --root data/02__italy_fwg_outputs --out data/03__italy_all_epw_DBT_streamlit --regional && python 05b_build_cti_epw_index_and_extract.py --root data/01__italy_cti/epw --out data/03__italy_all_epw_DBT_streamlit --regional && python 06_precompute_derived_stats.py --data-dir data/03__italy_all_epw_DBT_streamlit --process-cti --compute-aggregates
```

## For More Information

See the main [README.md](../../README.md) for:
- Complete workflow documentation
- Performance optimization tips
- Troubleshooting guide
- CTI data processing
- Regional vs per-station format explanation
- Data structure details
