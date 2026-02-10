#!/usr/bin/env python3
"""
Build CTI EPW index and extract hourly temperature data from CTI EPW files.

This script processes CTI (Italian Weather Stations) EPW files and creates:
1. cti_index.json - Metadata index of all CTI EPW files (location, lat/lon, etc.)
2. CTI__HR__<REGION>.parquet - Hourly DBT data per region (datetime-indexed)

Prerequisites:
- Run rename_cti_regions.ps1 first to rename files from CTI codes to standard codes
- CTI list CSV must exist: data/01__italy_cti/CTI__list__ITA_WeatherStations__All.csv

Usage:
    python 05b_build_cti_epw_index_and_extract.py --root data/01__italy_cti/epw --out data/03__italy_all_epw_DBT_streamlit --regional

Output files:
    - data/03__italy_all_epw_DBT_streamlit/cti_index.json
    - data/03__italy_all_epw_DBT_streamlit/CTI__HR__AB.parquet
    - data/03__italy_all_epw_DBT_streamlit/CTI__HR__BC.parquet
    - ... (one file per region)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd


EPW_COLUMNS = [
    "Year", "Month", "Day", "Hour", "Minute",
    "DataSourceUncertaintyFlags",
    "DryBulbTemperature", "DewPointTemperature", "RelativeHumidity",
    "AtmosphericStationPressure",
    "ExtraterrestrialHorizontalRadiation", "ExtraterrestrialDirectNormalRadiation",
    "HorizontalInfraredRadiationIntensity",
    "GlobalHorizontalRadiation", "DirectNormalRadiation", "DiffuseHorizontalRadiation",
    "GlobalHorizontalIlluminance", "DirectNormalIlluminance", "DiffuseHorizontalIlluminance",
    "ZenithLuminance",
    "WindDirection", "WindSpeed", "TotalSkyCover", "OpaqueSkyCover",
    "Visibility", "CeilingHeight", "PresentWeatherObservation", "PresentWeatherCodes",
    "PrecipitableWater", "AerosolOpticalDepth", "SnowDepth", "DaysSinceLastSnowfall",
    "Albedo", "LiquidPrecipitationDepth", "LiquidPrecipitationQuantity"
]


REGION_CODES = [
    "AB", "BC", "CM", "ER", "FV", "LB", "LG", "LM", "LZ", "MH",
    "ML", "PM", "PU", "SC", "SD", "TC", "TT", "UM", "VD", "VN"
]


@dataclass
class CTIRecord:
    """Metadata for one CTI EPW file."""
    rel_path: str           # Relative path from root (e.g., "AB__AQ__L'Aquila.epw")
    location_name: str      # Station name (e.g., "L'Aquila")
    location_id: str        # Station ID (e.g., "AB__AQ__L'Aquila")
    region: str             # Region code (e.g., "AB")
    province: str           # Province code (e.g., "AQ")
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    source: str = "CTI"     # Data source identifier


def load_cti_list(csv_path: Path) -> pd.DataFrame:
    """Load and parse the CTI station list CSV.
    
    Returns:
        DataFrame with columns: region, province, location, lat, lon, alt
    """
    # Try different encodings
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            
            # Check if we need semicolon delimiter
            if len(df.columns) == 1:
                df = pd.read_csv(csv_path, sep=";", encoding=encoding)
            
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Rename to standard names
            column_mapping = {
                "reg_shortname": "region",
                "location": "station_name",
                "lat": "latitude",
                "lon": "longitude",
                "alt": "altitude"
            }
            df = df.rename(columns=column_mapping)
            
            # Keep only needed columns
            needed_cols = ["region", "province", "station_name", "latitude", "longitude", "altitude"]
            available_cols = [c for c in needed_cols if c in df.columns]
            df = df[available_cols]
            
            print(f"  Loaded CTI list: {len(df)} stations (encoding: {encoding})")
            return df
            
        except Exception as e:
            continue
    
    raise ValueError(f"Could not load CTI list from {csv_path}")


def extract_station_info_from_filename(filename: str) -> Optional[Dict[str, str]]:
    """Extract region, province, and station name from CTI EPW filename.
    
    Expected format: AB__AQ__L'Aquila.epw or AB__AQ__Station Name.epw
    
    Returns:
        Dict with keys: region, province, station_name, location_id
    """
    # Remove .epw extension
    name = filename.replace(".epw", "").replace(".EPW", "")
    
    # Split by double underscore
    parts = name.split("__")
    
    if len(parts) >= 3:
        region = parts[0]
        province = parts[1]
        station_name = "__".join(parts[2:])  # In case station name has __
        
        return {
            "region": region,
            "province": province,
            "station_name": station_name,
            "location_id": f"{region}__{province}__{station_name}"
        }
    
    return None


def parse_epw_header(epw_path: Path) -> Optional[Dict[str, any]]:
    """Parse EPW header to extract location metadata.
    
    Returns:
        Dict with keys: city, state_province, country, latitude, longitude, elevation
    """
    try:
        with open(epw_path, "r", encoding="utf-8", errors="ignore") as f:
            # First line is LOCATION header
            location_line = f.readline().strip()
            
            if location_line.startswith("LOCATION"):
                parts = location_line.split(",")
                if len(parts) >= 10:
                    return {
                        "city": parts[1].strip(),
                        "state_province": parts[2].strip(),
                        "country": parts[3].strip(),
                        "data_source": parts[4].strip(),
                        "wmo_number": parts[5].strip(),
                        "latitude": float(parts[6]) if parts[6].strip() else None,
                        "longitude": float(parts[7]) if parts[7].strip() else None,
                        "timezone": float(parts[8]) if parts[8].strip() else None,
                        "elevation": float(parts[9]) if parts[9].strip() else None,
                    }
    except Exception as e:
        print(f"    Warning: Could not parse header from {epw_path.name}: {e}")
    
    return None


def read_epw_hourly_data(epw_path: Path) -> pd.DataFrame:
    """Read hourly data from EPW file, starting from line 9 (skipping 8 header lines).
    
    Returns:
        DataFrame with datetime index and DryBulbTemperature column (renamed to DBT)
    """
    try:
        # Skip 8 header lines, read CSV
        df = pd.read_csv(
            epw_path,
            skiprows=8,
            header=None,
            names=EPW_COLUMNS,
            na_values=["99.9", "999", "9999", "999999"],
        )
        
        # Create datetime
        df["datetime"] = pd.to_datetime(
            df[["Year", "Month", "Day", "Hour"]].astype(str).agg("-".join, axis=1) + ":00",
            format="%Y-%m-%d-%H:%M",
            errors="coerce"
        )
        
        # Keep only datetime and DBT
        df = df[["datetime", "DryBulbTemperature"]].copy()
        df = df.rename(columns={"DryBulbTemperature": "DBT"})
        
        # Drop rows with invalid datetime
        df = df.dropna(subset=["datetime"])
        
        # Set datetime as index
        df = df.set_index("datetime")
        
        return df
        
    except Exception as e:
        print(f"    Error reading hourly data from {epw_path.name}: {e}")
        return pd.DataFrame()


def build_cti_index(root: Path, cti_list_csv: Path, verbose: bool = False) -> List[CTIRecord]:
    """Scan CTI EPW files and build index.
    
    Args:
        root: Directory containing renamed CTI EPW files (e.g., AB__AQ__L'Aquila.epw)
        cti_list_csv: Path to CTI station list CSV
        verbose: Print detailed progress
    
    Returns:
        List of CTIRecord objects
    """
    print(f"\nScanning CTI EPW files in: {root}")
    
    # Load CTI list for lat/lon data
    cti_list = load_cti_list(cti_list_csv)
    
    # Find all EPW files
    epw_files = sorted(root.glob("*.epw"))
    print(f"  Found {len(epw_files)} EPW files")
    
    if len(epw_files) == 0:
        print(f"  Warning: No .epw files found. Did you run rename_cti_regions.ps1?")
        return []
    
    records = []
    
    for epw_path in epw_files:
        if verbose:
            print(f"  Processing: {epw_path.name}")
        
        # Extract info from filename
        file_info = extract_station_info_from_filename(epw_path.name)
        if not file_info:
            print(f"    Warning: Could not parse filename: {epw_path.name}")
            continue
        
        region = file_info["region"]
        province = file_info["province"]
        station_name = file_info["station_name"]
        location_id = file_info["location_id"]
        
        # Look up lat/lon from CTI list
        lat, lon, alt = None, None, None
        
        # Try to match by station name (case-insensitive, strip whitespace)
        matches = cti_list[
            (cti_list["region"] == region) &
            (cti_list["province"] == province)
        ]
        
        if not matches.empty:
            # Try exact match first
            exact_match = matches[matches["station_name"].str.lower().str.strip() == station_name.lower().strip()]
            if not exact_match.empty:
                row = exact_match.iloc[0]
                lat = row.get("latitude")
                lon = row.get("longitude")
                alt = row.get("altitude")
            else:
                # Take first match for region/province
                row = matches.iloc[0]
                lat = row.get("latitude")
                lon = row.get("longitude")
                alt = row.get("altitude")
        
        # If still missing, try to parse from EPW header
        if pd.isna(lat) or pd.isna(lon):
            header = parse_epw_header(epw_path)
            if header:
                if pd.isna(lat) and header.get("latitude"):
                    lat = header["latitude"]
                if pd.isna(lon) and header.get("longitude"):
                    lon = header["longitude"]
                if pd.isna(alt) and header.get("elevation"):
                    alt = header["elevation"]
        
        # Create record
        record = CTIRecord(
            rel_path=epw_path.name,
            location_name=station_name,
            location_id=location_id,
            region=region,
            province=province,
            latitude=float(lat) if pd.notna(lat) else None,
            longitude=float(lon) if pd.notna(lon) else None,
            altitude=float(alt) if pd.notna(alt) else None,
            source="CTI"
        )
        
        records.append(record)
        
        if verbose and (lat is None or lon is None):
            print(f"    Warning: Missing lat/lon for {station_name}")
    
    print(f"  Indexed {len(records)} CTI stations")
    
    # Show missing lat/lon count
    missing_coords = sum(1 for r in records if r.latitude is None or r.longitude is None)
    if missing_coords > 0:
        print(f"  ⚠️ {missing_coords} stations missing coordinates")
    
    return records


def extract_cti_hourly_regional(root: Path, records: List[CTIRecord], output_dir: Path, verbose: bool = False):
    """Extract hourly DBT data and save as regional parquet files.
    
    Creates: CTI__HR__AB.parquet, CTI__HR__BC.parquet, etc.
    Each file contains: datetime (index), DBT, location_id columns
    """
    print(f"\nExtracting hourly data to regional parquet files...")
    
    # Group records by region
    regional_data = {code: [] for code in REGION_CODES}
    
    for record in records:
        if verbose:
            print(f"  Reading: {record.rel_path}")
        
        epw_path = root / record.rel_path
        hourly_df = read_epw_hourly_data(epw_path)
        
        if hourly_df.empty:
            print(f"    Warning: No valid hourly data in {record.rel_path}")
            continue
        
        # Add location_id column
        hourly_df["location_id"] = record.location_id
        
        # Add to regional buffer
        if record.region in regional_data:
            regional_data[record.region].append(hourly_df)
        
        if verbose:
            print(f"    Extracted {len(hourly_df):,} hourly rows")
    
    # Write regional files
    files_written = 0
    total_rows = 0
    
    for region_code in REGION_CODES:
        if not regional_data[region_code]:
            continue
        
        # Combine all stations in this region
        region_df = pd.concat(regional_data[region_code], axis=0)
        
        # Sort by datetime and location_id
        region_df = region_df.sort_index()
        
        # Save to parquet
        output_file = output_dir / f"CTI__HR__{region_code}.parquet"
        region_df.to_parquet(output_file, index=True)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  ✓ {output_file.name}: {len(region_df):,} rows, {file_size_mb:.2f} MB")
        
        files_written += 1
        total_rows += len(region_df)
    
    print(f"\n  Total: {files_written} regional files, {total_rows:,} hourly rows")


def main():
    ap = argparse.ArgumentParser(
        description="Build CTI EPW index and extract hourly temperature data"
    )
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing CTI EPW files (after rename_cti_regions.ps1)"
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for index and parquet files"
    )
    ap.add_argument(
        "--cti-list",
        type=str,
        default=None,
        help="Path to CTI station list CSV (default: auto-detect from root/../CTI__list__ITA_WeatherStations__All.csv)"
    )
    ap.add_argument(
        "--regional",
        action="store_true",
        help="Create regional parquet files (CTI__HR__XX.parquet)"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    args = ap.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent  # Go up to FWG root
    root_path = (script_dir / args.root).resolve()
    output_dir = (script_dir / args.out).resolve()
    
    # Auto-detect CTI list path if not provided
    if args.cti_list:
        cti_list_path = (script_dir / args.cti_list).resolve()
    else:
        # Try ../CTI__list__ITA_WeatherStations__All.csv relative to root
        cti_list_path = root_path.parent / "CTI__list__ITA_WeatherStations__All.csv"
    
    # Validate paths
    if not root_path.exists():
        print(f"❌ Root directory not found: {root_path}")
        print(f"   Did you run rename_cti_regions.ps1 first?")
        sys.exit(1)
    
    if not cti_list_path.exists():
        print(f"❌ CTI list CSV not found: {cti_list_path}")
        print(f"   Expected: data/01__italy_cti/CTI__list__ITA_WeatherStations__All.csv")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CTI EPW Index Builder and Data Extractor")
    print("="*70)
    print(f"Root: {root_path}")
    print(f"CTI List: {cti_list_path}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'Regional' if args.regional else 'Index only'}")
    
    # Build index
    records = build_cti_index(root_path, cti_list_path, verbose=args.verbose)
    
    if not records:
        print("\n❌ No CTI records found. Exiting.")
        sys.exit(1)
    
    # Save index
    index_path = output_dir / "cti_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved CTI index: {index_path.name} ({len(records)} records)")
    
    # Extract regional data if requested
    if args.regional:
        extract_cti_hourly_regional(root_path, records, output_dir, verbose=args.verbose)
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 06_precompute_derived_stats.py with --process-cti flag to include CTI in daily stats")
    print("2. Or run the main 05_build_epw_index_and_extract.py for TMYx data")
    print()


if __name__ == "__main__":
    main()
