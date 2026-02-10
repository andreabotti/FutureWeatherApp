#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


ITALY_INDEX_URL = "https://climate.onebuilding.org/WMO_Region_6_Europe/ITA_Italy/index.html"


@dataclass(frozen=True)
class DownloadItem:
    url: str
    filename: str


def parse_italy_zip_links(index_url: str) -> list[DownloadItem]:
    """
    Scrape the Italy index page and collect all *.zip links.
    """
    r = requests.get(index_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    items: list[DownloadItem] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().endswith(".zip"):
            continue

        full_url = urljoin(index_url, href)
        fname = Path(urlparse(full_url).path).name
        if not fname:
            continue

        items.append(DownloadItem(url=full_url, filename=fname))

    # de-duplicate by URL
    uniq = {it.url: it for it in items}
    return sorted(uniq.values(), key=lambda x: x.filename.lower())


def download_file(url: str, out_path: Path, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)


def extract_epw_from_zip(zip_path: Path, dest_dir: Path, overwrite: bool = False) -> list[Path]:
    """
    Extract only *.epw from a climate.onebuilding zip.
    Returns list of extracted EPW paths.
    """
    extracted: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if m.lower().endswith(".epw")]

        for member in members:
            # Flatten internal folders; keep the EPW filename only
            epw_name = Path(member).name
            out_path = dest_dir / epw_name

            if out_path.exists() and not overwrite:
                extracted.append(out_path)
                continue

            with z.open(member) as src, out_path.open("wb") as dst:
                dst.write(src.read())

            extracted.append(out_path)

    return extracted


def main() -> None:
    p = argparse.ArgumentParser(description="Download all Italy climate.onebuilding ZIPs and extract EPW files.")
    p.add_argument("--out", required=True, help="Output root folder (downloads + extracted EPWs).")
    p.add_argument("--index", default=ITALY_INDEX_URL, help="Italy index URL (default: official Italy page).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads/extracted EPWs.")
    args = p.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    zips_dir = out_root / "zips"
    epw_dir = out_root / "epw"

    print(f"Scraping: {args.index}")
    items = parse_italy_zip_links(args.index)
    if not items:
        print("No ZIP links found. The page structure may have changed.")
        sys.exit(1)

    print(f"Found {len(items)} ZIP(s). Downloading to: {zips_dir}")
    for i, it in enumerate(items, 1):
        zip_path = zips_dir / it.filename
        print(f"[{i}/{len(items)}] {it.filename}")
        download_file(it.url, zip_path, overwrite=args.overwrite)

    # Extract EPWs (organise by Italian region code in the filename, e.g. ITA_ER_..., ITA_LM_..., etc.)
    print(f"\nExtracting EPWs to: {epw_dir}")
    extracted_total = 0
    for zip_path in sorted(zips_dir.glob("*.zip")):
        # Try to classify by region prefix in filename: ITA_<REGION>_...
        m = re.match(r"^ITA_([A-Z]{2})_", zip_path.stem)
        region_code = m.group(1) if m else "UNK"
        dest = epw_dir / region_code

        epws = extract_epw_from_zip(zip_path, dest, overwrite=args.overwrite)
        extracted_total += len(epws)

    print(f"Done. Extracted EPWs: {extracted_total}")


if __name__ == "__main__":
    main()
