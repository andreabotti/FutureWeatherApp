#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_MODELS = "ICHEC_EC_EARTH_DMI_HIRHAM5,MOHC_HadGEM2_ES_SMHI_RCA4"


def iter_epws(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.epw") if p.is_file())


def run_fwg(java_exe: str, jar_path: Path, epw_path: Path, output_folder: Path, models_csv: str, verbose: bool) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        java_exe, "-jar", str(jar_path),
        f"-epw={epw_path}",
        f"-output_folder={output_folder}",
        f"-models={models_csv}",
        "-ensemble=true",
    ]
    if verbose:
        print("CMD:", " ".join(cmd))

    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--jar", dest="jar_path", required=True)
    p.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model tokens for -models=")
    p.add_argument("--java", dest="java_exe", default="java")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    in_root = Path(args.in_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    jar_path = Path(args.jar_path).expanduser().resolve()

    epws = iter_epws(in_root)
    if not epws:
        print(f"No EPWs found under: {in_root}")
        return

    for i, epw in enumerate(epws, 1):
        rel_parent = epw.parent.relative_to(in_root)
        per_epw_out = out_root / rel_parent / epw.stem
        print(f"[{i}/{len(epws)}] {epw}")
        run_fwg(args.java_exe, jar_path, epw, per_epw_out, args.models, args.verbose)

    print("Done.")


if __name__ == "__main__":
    main()
