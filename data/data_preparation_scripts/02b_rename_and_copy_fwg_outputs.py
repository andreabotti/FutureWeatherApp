#!/usr/bin/env python3
"""
Rename FWG morphed EPW/STAT files under 02__italy_fwg_outputs so that filenames
include the baseline folder name (including station_id and TMYx scenario),
then copy them into 03__italy_epw_all/<REGION>/.

IMPORTANT FIX vs previous version:
- Recursively scans inside each station folder (EPWs are often in /output or similar).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable


FWG_MORPH_RE = re.compile(
    r"""
    ^(?P<anyprefix>.+?)_                # anything up to first underscore
    (?P<model>[^_]+)_                   # model token (often Ensemble)
    (?P<scenario>(?:rcp|ssp)\d+)_       # rcp26 / rcp45 / rcp85 / ssp126 etc
    (?P<year>\d{4})                     # 2050 / 2080 etc
    (?P<ext>\.(?:epw|stat))$            # .epw or .stat
    """,
    re.IGNORECASE | re.VERBOSE,
)

ALREADY_RENAMED_RE = re.compile(
    r"""^.+__[^_]+__(?:rcp|ssp)\d+__\d{4}\.(?:epw|stat)$""",
    re.IGNORECASE,
)


@dataclass
class ActionRow:
    action: str              # rename/copy/skip/error
    region: str
    baseline_folder: str
    src_path: str
    dst_path: str
    reason: str


def _is_baseline_epw(file_path: Path, baseline_folder_name: str) -> bool:
    return file_path.name.lower() == f"{baseline_folder_name}.epw".lower()


def _is_baseline_stat(file_path: Path, baseline_folder_name: str) -> bool:
    return file_path.name.lower() == f"{baseline_folder_name}.stat".lower()


def _parse_morph_filename(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Return (model, scenario, year, ext) from a morph filename.
    """
    m = FWG_MORPH_RE.match(name)
    if not m:
        return None
    model = m.group("model").strip()
    scenario = m.group("scenario").strip().lower()
    year = m.group("year").strip()
    ext = m.group("ext").lower()
    return model, scenario, year, ext


def _build_new_name(baseline_folder: str, model: str, scenario: str, year: str, ext: str) -> str:
    return f"{baseline_folder}__{model}__{scenario}__{year}{ext}"


def _safe_rename(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> Tuple[bool, str]:
    if src == dst:
        return False, "source and destination are the same"
    if dst.exists():
        if not overwrite:
            return False, "destination exists (use --overwrite)"
        if not dry_run:
            dst.unlink()
    if not dry_run:
        src.rename(dst)
    return True, "renamed"


def _safe_copy(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> Tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return False, "destination exists (use --overwrite)"
        if not dry_run:
            dst.unlink()
    if not dry_run:
        shutil.copy2(src, dst)
    return True, "copied"


def iter_region_station_folders(fwg_root: Path) -> Iterable[Tuple[str, Path]]:
    for region_dir in sorted([p for p in fwg_root.iterdir() if p.is_dir()]):
        region_code = region_dir.name
        for station_dir in sorted([p for p in region_dir.iterdir() if p.is_dir()]):
            yield region_code, station_dir


def iter_files_recursive(station_dir: Path, include_stat: bool) -> Iterable[Path]:
    exts = {".epw"}
    if include_stat:
        exts.add(".stat")

    # Traverse everything under the station folder (including /output)
    for p in station_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwg-root", required=True)
    ap.add_argument("--epw-all-out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-stat", action="store_true")
    ap.add_argument("--copy-baseline", action="store_true")
    ap.add_argument("--report-csv", default="rename_copy_report.csv")
    args = ap.parse_args()

    fwg_root = Path(args.fwg_root).resolve()
    out_root = Path(args.epw_all_out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    include_stat = not args.no_stat
    rows: list[ActionRow] = []

    print("=" * 72)
    print("FWG rename + copy (RECURSIVE)")
    print("=" * 72)
    print(f"FWG root      : {fwg_root}")
    print(f"EPW all out   : {out_root}")
    print(f"dry-run       : {args.dry_run}")
    print(f"overwrite     : {args.overwrite}")
    print(f"include .stat : {include_stat}")
    print(f"copy baseline : {args.copy_baseline}")
    print("=" * 72)

    if not fwg_root.exists():
        raise SystemExit(f"--fwg-root does not exist: {fwg_root}")

    total_found = 0

    for region, station_dir in iter_region_station_folders(fwg_root):
        baseline_folder = station_dir.name
        dest_region_dir = out_root / region
        dest_region_dir.mkdir(parents=True, exist_ok=True)

        files = list(iter_files_recursive(station_dir, include_stat))
        total_found += len(files)

        for f in sorted(files):
            # Baseline files
            if _is_baseline_epw(f, baseline_folder) or _is_baseline_stat(f, baseline_folder):
                if args.copy_baseline:
                    dst = dest_region_dir / f.name
                    ok, reason = _safe_copy(f, dst, args.overwrite, args.dry_run)
                    rows.append(ActionRow("copy" if ok else "skip", region, baseline_folder, str(f), str(dst), reason))
                    print(f"[{region}] baseline {'COPY' if ok else 'SKIP'}: {f.name} -> {dst.name} | {reason}")
                else:
                    rows.append(ActionRow("skip", region, baseline_folder, str(f), "", "baseline (not copied)"))
                continue

            # Already renamed?
            if ALREADY_RENAMED_RE.match(f.name):
                dst = dest_region_dir / f.name
                ok, reason = _safe_copy(f, dst, args.overwrite, args.dry_run)
                rows.append(ActionRow("copy" if ok else "skip", region, baseline_folder, str(f), str(dst), reason))
                print(f"[{region}] already-renamed {'COPY' if ok else 'SKIP'}: {f.name} -> {dst.name} | {reason}")
                continue

            parsed = _parse_morph_filename(f.name)
            if not parsed:
                rows.append(ActionRow("skip", region, baseline_folder, str(f), "", "does not match FWG morph pattern"))
                continue

            model, scenario, year, ext = parsed
            new_name = _build_new_name(baseline_folder, model, scenario, year, ext)
            new_path = f.with_name(new_name)

            ok_rename, reason_rename = _safe_rename(f, new_path, args.overwrite, args.dry_run)
            rows.append(ActionRow("rename" if ok_rename else "skip", region, baseline_folder, str(f), str(new_path), reason_rename))
            print(f"[{region}] {'RENAME' if ok_rename else 'SKIP'}: {f.name} -> {new_name} | {reason_rename}")

            # Copy to 03__italy_epw_all/<REGION>/
            copy_src = new_path if (not args.dry_run or ok_rename) else f
            dst = dest_region_dir / new_name
            ok_copy, reason_copy = _safe_copy(copy_src, dst, args.overwrite, args.dry_run)
            rows.append(ActionRow("copy" if ok_copy else "skip", region, baseline_folder, str(copy_src), str(dst), reason_copy))
            print(f"[{region}] {'COPY' if ok_copy else 'SKIP'}: {copy_src.name} -> {dst.name} | {reason_copy}")

    report_path = out_root / args.report_csv
    with open(report_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["action", "region", "baseline_folder", "src_path", "dst_path", "reason"])
        for r in rows:
            w.writerow([r.action, r.region, r.baseline_folder, r.src_path, r.dst_path, r.reason])

    print("=" * 72)
    print(f"Total station files found (epw/stat): {total_found}")
    print(f"Report written: {report_path}")
    print(f"Total actions : {len(rows)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
