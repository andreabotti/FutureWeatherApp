#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Collect EPWs into a single folder (optionally mirroring subfolders).")
    p.add_argument("--in", dest="in_dir", required=True, help="Folder to search for *.epw (recursive).")
    p.add_argument("--out", dest="out_dir", required=True, help="Output folder.")
    p.add_argument("--mirror", action="store_true", help="Mirror the input folder structure under --out.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in --out.")
    args = p.parse_args()

    in_root = Path(args.in_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    epws = sorted([p for p in in_root.rglob("*.epw") if p.is_file()])
    if not epws:
        print(f"No EPWs found under: {in_root}")
        return

    for i, epw in enumerate(epws, 1):
        if args.mirror:
            rel = epw.relative_to(in_root)
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest = out_root / epw.name

        if dest.exists() and not args.overwrite:
            continue

        shutil.copy2(epw, dest)
        print(f"[{i}/{len(epws)}] {epw} -> {dest}")

    print("Done.")


if __name__ == "__main__":
    main()
