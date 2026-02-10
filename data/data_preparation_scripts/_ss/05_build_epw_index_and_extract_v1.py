#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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


@dataclass
class EPWIndexRecord:
    scenario: str
    rel_path: str
    abs_path: str
    filename: str
    size_bytes: int
    mtime_iso: str
    header_location_line: str
    location_name: Optional[str]
    country: Optional[str]
    wmo: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[float]
    elevation_m: Optional[float]


def discover_epw_files(root_dir: Path) -> List[Path]:
    """
    Case-insensitive EPW discovery.
    Also finds .EPW. (Not .epw.gz — add if you need it.)
    """
    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".epw":
            files.append(p)
    return sorted(files)


def scenario_from_path(epw_path: Path, root_dir: Path) -> str:
    """
    Default scenario label = parent folder name.
    Adjust if your FWG structure encodes scenario deeper.
    """
    rel = epw_path.resolve().relative_to(root_dir.resolve())
    return rel.parent.name


def read_epw_header(epw_path: Path) -> List[str]:
    text = epw_path.read_text(encoding="utf-8", errors="ignore")
    return text.splitlines()[:8]


def parse_location_from_header(line: str) -> Tuple[Optional[str], Optional[str], Optional[str],
                                                  Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not line:
        return (None, None, None, None, None, None, None)

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 10 or parts[0].upper() != "LOCATION":
        return (None, None, None, None, None, None, None)

    city = parts[1] if len(parts) > 1 else None
    country = parts[3] if len(parts) > 3 else None
    wmo = parts[5] if len(parts) > 5 else None

    def to_float(x: str) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    lat = to_float(parts[6]) if len(parts) > 6 else None
    lon = to_float(parts[7]) if len(parts) > 7 else None
    tz = to_float(parts[8]) if len(parts) > 8 else None
    elev = to_float(parts[9]) if len(parts) > 9 else None

    return (city, country, wmo, lat, lon, tz, elev)


def parse_epw_dbt_rh(epw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        epw_path,
        skiprows=8,
        header=None,
        names=EPW_COLUMNS,
        na_values=["", "NA", "N/A", "-999", "-999.0"],
        engine="python"
    )

    # EPW Hour is 1-24; convert to 0-23
    hour = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int) - 1
    hour = hour.clip(lower=0, upper=23)

    dt = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["Year"], errors="coerce"),
            month=pd.to_numeric(df["Month"], errors="coerce"),
            day=pd.to_numeric(df["Day"], errors="coerce"),
            hour=hour
        ),
        errors="coerce"
    )

    out = pd.DataFrame(
        {
            "DBT": pd.to_numeric(df["DryBulbTemperature"], errors="coerce"),
            "RH": pd.to_numeric(df["RelativeHumidity"], errors="coerce"),
        },
        index=dt
    )

    out.index.name = "datetime"
    out = out.dropna(subset=["datetime"], how="any") if "datetime" in out.columns else out
    out = out[~out.index.isna()].sort_index()
    return out


def write_scan_summary(epw_files: List[Path], root_dir: Path, out_dir: Path) -> None:
    rows = []
    for p in epw_files:
        rel = p.resolve().relative_to(root_dir.resolve())
        top = rel.parts[0] if len(rel.parts) else ""
        rows.append({"top_level": top, "rel_path": str(rel), "abs_path": str(p.resolve())})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scan_summary.csv", index=False)

    # Print counts per top-level folder
    if not df.empty:
        counts = df["top_level"].value_counts().reset_index()
        counts.columns = ["top_level", "count"]
        print("\nEPW count by top-level folder:")
        print(counts.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder to scan for EPW files (e.g. ./data/02__italy_fwg_outputs)")
    ap.add_argument("--out", required=True, help="Output folder (e.g. ./data/03__epw_extract)")
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet", help="Output format")
    ap.add_argument("--wide", action="store_true", help="Also write a wide table (one column per scenario)")
    ap.add_argument("--dry-run", action="store_true", help="Only scan + write scan_summary.csv, no parsing")
    args = ap.parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    print(f"ROOT: {root_dir}")
    print(f"OUT:  {out_dir}")

    if not root_dir.exists():
        raise SystemExit(f"Root folder does not exist: {root_dir}")

    epw_files = discover_epw_files(root_dir)
    print(f"Discovered EPWs: {len(epw_files)}")

    if not epw_files:
        raise SystemExit("No .epw/.EPW files found. Check --root and extensions.")

    write_scan_summary(epw_files, root_dir, out_dir)

    if args.dry_run:
        print("\nDry run complete (no parsing).")
        return

    index_records: List[EPWIndexRecord] = []
    tidy_parts: List[pd.DataFrame] = []

    for i, p in enumerate(epw_files, start=1):
        if i % 50 == 0 or i == 1 or i == len(epw_files):
            print(f"Parsing {i}/{len(epw_files)}: {p.name}")

        st = p.stat()
        header = read_epw_header(p)
        loc_line = header[0] if header else ""
        city, country, wmo, lat, lon, tz, elev = parse_location_from_header(loc_line)

        scenario = scenario_from_path(p, root_dir)
        rel = str(p.resolve().relative_to(root_dir))

        index_records.append(
            EPWIndexRecord(
                scenario=scenario,
                rel_path=rel,
                abs_path=str(p.resolve()),
                filename=p.name,
                size_bytes=st.st_size,
                mtime_iso=pd.to_datetime(st.st_mtime, unit="s").isoformat(),
                header_location_line=loc_line,
                location_name=city,
                country=country,
                wmo=wmo,
                latitude=lat,
                longitude=lon,
                timezone=tz,
                elevation_m=elev,
            )
        )

        df = parse_epw_dbt_rh(p).reset_index()
        df["scenario"] = scenario
        df["rel_path"] = rel
        tidy_parts.append(df)

    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON index
    idx_path = out_dir / "epw_index.json"
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in index_records], f, indent=2, ensure_ascii=False)
    print(f"Wrote index: {idx_path}")

    # Tidy output
    tidy = pd.concat(tidy_parts, ignore_index=True)
    tidy["datetime"] = pd.to_datetime(tidy["datetime"], errors="coerce")
    tidy = tidy.dropna(subset=["datetime"]).sort_values(["scenario", "datetime"])

    if args.format in ("parquet", "both"):
        tidy.to_parquet(out_dir / "dbt_rh_tidy.parquet", index=False)
        print(f"Wrote: {out_dir / 'dbt_rh_tidy.parquet'}")
    if args.format in ("csv", "both"):
        tidy.to_csv(out_dir / "dbt_rh_tidy.csv", index=False)
        print(f"Wrote: {out_dir / 'dbt_rh_tidy.csv'}")

    # Optional wide
    if args.wide:
        wide = tidy.set_index(["datetime", "scenario"])[["DBT", "RH"]].unstack("scenario").sort_index()
        wide.columns = [f"{metric}__{scenario}" for metric, scenario in wide.columns]
        wide = wide.reset_index()

        if args.format in ("parquet", "both"):
            wide.to_parquet(out_dir / "dbt_rh_wide.parquet", index=False)
            print(f"Wrote: {out_dir / 'dbt_rh_wide.parquet'}")
        if args.format in ("csv", "both"):
            wide.to_csv(out_dir / "dbt_rh_wide.csv", index=False)
            print(f"Wrote: {out_dir / 'dbt_rh_wide.csv'}")


if __name__ == "__main__":
    main()
