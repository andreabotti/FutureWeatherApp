#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

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
    source: Optional[str]
    region: Optional[str]
    group: Optional[str]
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
    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".epw":
            # Skip common FWG output helper folders that may contain duplicates
            # or non-primary EPWs.
            parts_lc = {x.lower() for x in p.parts}
            if "output" in parts_lc:
                continue
            files.append(p)
    return sorted(files)


def rel_to_project(epw_path: Path, project_root: Path, fallback_root: Path) -> str:
    """
    Prefer making rel paths relative to the project root (so multiple roots can be merged).
    If the file is outside the project root, fall back to a synthetic path that is still unique.
    """
    epw_abs = epw_path.resolve()
    project_root = project_root.resolve()
    fallback_root = fallback_root.resolve()
    try:
        return str(epw_abs.relative_to(project_root))
    except Exception:
        return str(Path(fallback_root.name) / epw_abs.relative_to(fallback_root))


def scenario_from_rel(rel_path: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Heuristic scenario labeling from a project-relative path.
    Expects layouts like:
      data/01__italy_epw_all/<REGION>/<file>.epw
      data/02__italy_fwg_outputs/<REGION>/<GROUP>/<file>.epw
    """
    parts = Path(rel_path).parts
    source = parts[0] if len(parts) >= 1 else None
    region = parts[1] if len(parts) >= 2 else None
    group = parts[2] if len(parts) >= 3 else None
    scenario = "/".join([p for p in (source, region) if p]) or (group or "unknown")
    return scenario, source, region, group


def read_epw_header(epw_path: Path) -> List[str]:
    text = epw_path.read_text(encoding="utf-8", errors="ignore")
    return text.splitlines()[:8]


def parse_location_from_header(line: str) -> Tuple[Optional[str], Optional[str], Optional[str],
                                                  Optional[float], Optional[float], Optional[float], Optional[float]]:
    parts = [p.strip() for p in (line or "").split(",")]
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

    hour = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int) - 1
    hour = hour.clip(0, 23)

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
            "datetime": dt,
            "DBT": pd.to_numeric(df["DryBulbTemperature"], errors="coerce"),
            "RH": pd.to_numeric(df["RelativeHumidity"], errors="coerce"),
        }
    ).dropna(subset=["datetime"])

    return out


def append_csv(path: Path, df: pd.DataFrame) -> None:
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        action="append",
        help="One or more input roots. Pass multiple times to merge sources (e.g. --root data/01__italy_epw_all --root data/02__italy_fwg_outputs).",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet")
    ap.add_argument("--chunk", type=int, default=100, help="Write outputs every N files")
    ap.add_argument("--limit", type=int, default=0, help="For testing: max EPWs to parse (0 = no limit)")
    args = ap.parse_args()

    project_root = Path(".").resolve()
    roots = [Path(r).expanduser().resolve() for r in args.root]
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        raise SystemExit("Missing root(s):\n- " + "\n- ".join(missing))
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("ROOTS:")
    for r in roots:
        print(f"  - {r}")
    print(f"OUT:  {out_dir}")

    epw_files: List[Path] = []
    for r in roots:
        epw_files.extend(discover_epw_files(r))
    # Deduplicate (same absolute path could be discovered via overlapping roots)
    epw_files = sorted({p.resolve() for p in epw_files})
    if args.limit and args.limit > 0:
        epw_files = epw_files[: args.limit]

    total = len(epw_files)
    print(f"Discovered EPWs: {total}")
    if total == 0:
        raise SystemExit("No EPW files found.")

    # Outputs
    idx_path = out_dir / "epw_index.json"
    tidy_parquet_path = out_dir / "dbt_rh_tidy.parquet"
    tidy_csv_path = out_dir / "dbt_rh_tidy.csv"
    log_csv = out_dir / "parse_log.csv"
    failed_txt = out_dir / "failed_files.txt"

    index_records: List[EPWIndexRecord] = []
    tidy_buffer: List[pd.DataFrame] = []
    log_rows = []
    failed = []

    # If re-running, remove previous chunked outputs for cleanliness
    # (comment out if you prefer appending)
    for p in [tidy_parquet_path, tidy_csv_path, log_csv, failed_txt]:
        if p.exists():
            p.unlink()

    for i, p in enumerate(epw_files, start=1):
        # Pick a fallback root for synthetic rel paths (first root that contains the file)
        fallback_root = next((r for r in roots if str(p).startswith(str(r))), roots[0])
        rel = rel_to_project(p, project_root=project_root, fallback_root=fallback_root)
        scenario, source, region, group = scenario_from_rel(rel)

        try:
            st = p.stat()
            header = read_epw_header(p)
            loc_line = header[0] if header else ""
            city, country, wmo, lat, lon, tz, elev = parse_location_from_header(loc_line)

            index_records.append(
                EPWIndexRecord(
                    scenario=scenario,
                    source=source,
                    region=region,
                    group=group,
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

            df = parse_epw_dbt_rh(p)
            df["scenario"] = scenario
            df["rel_path"] = rel
            tidy_buffer.append(df)

            log_rows.append({"rel_path": rel, "scenario": scenario, "status": "OK", "error": ""})

        except Exception as e:
            err = repr(e)
            failed.append(rel)
            log_rows.append({"rel_path": rel, "scenario": scenario, "status": "FAIL", "error": err})

        # Progress
        if i == 1 or i % 50 == 0 or i == total:
            ok = sum(1 for r in log_rows if r["status"] == "OK")
            fail = sum(1 for r in log_rows if r["status"] == "FAIL")
            print(f"[{i}/{total}] OK={ok} FAIL={fail}  (last: {p.name})")

        # Chunk write
        if (i % args.chunk == 0) or (i == total):
            if tidy_buffer:
                tidy_chunk = pd.concat(tidy_buffer, ignore_index=True)
                tidy_chunk["datetime"] = pd.to_datetime(tidy_chunk["datetime"], errors="coerce")
                tidy_chunk = tidy_chunk.dropna(subset=["datetime"])

                if args.format in ("csv", "both"):
                    append_csv(tidy_csv_path, tidy_chunk)

                if args.format in ("parquet", "both"):
                    # Parquet append is not trivial without pyarrow dataset;
                    # easiest: keep one big parquet at end, or write chunked parquet files.
                    # We'll write chunked parquet files and combine later if needed.
                    part = out_dir / f"dbt_rh_tidy_part_{i:05d}.parquet"
                    tidy_chunk.to_parquet(part, index=False)

                tidy_buffer.clear()

            # log chunk
            if log_rows:
                log_df = pd.DataFrame(log_rows)
                append_csv(log_csv, log_df)
                log_rows.clear()

    # Final write index JSON
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in index_records], f, indent=2, ensure_ascii=False)
    print(f"Wrote index: {idx_path}")

    # Failed list
    if failed:
        failed_txt.write_text("\n".join(failed), encoding="utf-8")
        print(f"Some files failed. See: {failed_txt}")
    else:
        print("No parse failures")

    # If parquet was requested, merge chunked parquet parts into one final parquet
    if args.format in ("parquet", "both"):
        parts = sorted(out_dir.glob("dbt_rh_tidy_part_*.parquet"))
        if parts:
            print(f"Merging {len(parts)} parquet parts...")
            df_all = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            df_all.to_parquet(tidy_parquet_path, index=False)
            print(f"Wrote: {tidy_parquet_path}")

            # Optional: remove parts after merge
            for p in parts:
                p.unlink()

    if args.format in ("csv", "both"):
        print(f"Wrote: {tidy_csv_path}")
    print(f"Wrote parse log: {log_csv}")


if __name__ == "__main__":
    main()
