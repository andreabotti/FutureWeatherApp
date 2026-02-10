#!/usr/bin/env python3
"""
Copy baseline EPW files from:
  data/01__italy_epw_all/<REGION>/*.epw
to:
  data/03__italy_epw_all/<REGION>/*.epw

Preserves region folder structure (AB, BC, ...).

Options:
  --dry-run     : do not copy, only log
  --overwrite   : overwrite files in destination if they exist
  --include-stat: also copy .stat files
  --report-csv  : write report CSV into destination root

No temp folders, no _tmp_parts.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Row:
    ts: str
    action: str     # COPY / SKIP / ERROR
    region: str
    src: str
    dst: str
    reason: str


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def iter_files(src_root: Path, include_stat: bool) -> Iterable[Path]:
    exts = {".epw"}
    if include_stat:
        exts.add(".stat")
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Baseline EPW root (e.g. data/01__italy_epw_all)")
    ap.add_argument("--dst", required=True, help="Destination root (e.g. data/03__italy_epw_all)")
    ap.add_argument("--dry-run", action="store_true", help="Do not copy; only log actions.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist.")
    ap.add_argument("--include-stat", action="store_true", help="Also copy .stat files.")
    ap.add_argument("--report-csv", default="baseline_copy_report.csv", help="Report CSV filename (written into --dst).")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        raise SystemExit(f"Source folder not found: {src_root}")

    rows: list[Row] = []
    copied = skipped = errors = 0

    print("=" * 72)
    print("Copy baseline EPWs")
    print("=" * 72)
    print(f"Source   : {src_root}")
    print(f"Dest     : {dst_root}")
    print(f"dry-run  : {args.dry_run}")
    print(f"overwrite: {args.overwrite}")
    print(f"stat     : {args.include_stat}")
    print("=" * 72)

    for src in iter_files(src_root, include_stat=args.include_stat):
        try:
            rel = src.relative_to(src_root)
            # Expecting: <REGION>/<filename>
            if len(rel.parts) < 2:
                skipped += 1
                rows.append(Row(now(), "SKIP", "NA", str(src), "", "unexpected path (no region folder)"))
                continue

            region = rel.parts[0]
            dst = dst_root / region / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists() and not args.overwrite:
                skipped += 1
                rows.append(Row(now(), "SKIP", region, str(src), str(dst), "exists"))
                continue

            if not args.dry_run:
                if dst.exists() and args.overwrite:
                    dst.unlink()
                shutil.copy2(src, dst)

            copied += 1
            rows.append(Row(now(), "COPY", region, str(src), str(dst), "copied" if not args.dry_run else "dry-run"))

        except Exception as e:
            errors += 1
            rows.append(Row(now(), "ERROR", "NA", str(src), "", str(e)))

    # Write report
    report_path = dst_root / args.report_csv
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "action", "region", "src", "dst", "reason"])
        for r in rows:
            w.writerow([r.ts, r.action, r.region, r.src, r.dst, r.reason])

    print("=" * 72)
    print(f"Copied : {copied}")
    print(f"Skipped: {skipped}")
    print(f"Errors : {errors}")
    print(f"Report : {report_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
