"""
06E_precompute_geodata.py
─────────────────────────────────────────────────────────────────────────────
Downloads and saves the three GeoJSON files used by all D3 dashboards into
  data/geo/
      italy_outline.geojson   (~50 KB  — Italy polygon, extracted from world-atlas)
      regions.geojson         (~55 KB  — 20 regioni, simplified)
      provinces.geojson       (~210 KB — 107 province, simplified)

Run ONCE after setup.  The app reads from disk on startup (st.cache_resource)
and injects them inline into every D3 iframe, eliminating 3 sequential network
fetches (≈ 6 MB / 2-5 s) on every first render.

Simplification
──────────────
If `mapshaper` (https://github.com/mbloch/mapshaper) is on your PATH the
script automatically runs:
    mapshaper INPUT.geojson -simplify 1% keep-shapes -o format=geojson OUT.geojson

Without mapshaper the raw GeoJSON is saved (still a win vs. network fetch).

Usage
─────
    python data/data_preparation_scripts/06E_precompute_geodata.py
    python data/data_preparation_scripts/06E_precompute_geodata.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
GEO_DIR = SCRIPT_DIR.parent.parent / "data" / "geo"

URLS = {
    "world_atlas":  "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json",
    "regions":      "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson",
    "provinces":    "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_provinces.geojson",
}

ITALY_COUNTRY_ID = 380   # ISO numeric id used in world-atlas


# ── minimal TopoJSON decoder ──────────────────────────────────────────────────
def _decode_topo_arcs(raw_arcs: list, scale: list, translate: list) -> list:
    """Delta-decode quantized TopoJSON arcs → list of [[lon, lat], ...] rings."""
    decoded = []
    for arc in raw_arcs:
        ring: list = []
        x = y = 0
        for pt in arc:
            x += pt[0]
            y += pt[1]
            ring.append([x * scale[0] + translate[0], y * scale[1] + translate[1]])
        decoded.append(ring)
    return decoded


def _stitch_ring(arc_indices: list, arcs: list) -> list:
    """Stitch a list of arc indices (positive = forward, negative = reversed)."""
    ring: list = []
    for idx in arc_indices:
        if idx >= 0:
            coords = arcs[idx]
        else:
            coords = list(reversed(arcs[~idx]))
        # Arcs share endpoints — append all except the last (= first of next arc)
        ring.extend(coords[:-1] if ring else coords)
    return ring


def _decode_geometry(geom: dict, arcs: list) -> dict | None:
    gtype = geom.get("type")
    if gtype == "Polygon":
        coords = [_stitch_ring(ring, arcs) for ring in geom["arcs"]]
        return {"type": "Polygon", "coordinates": coords}
    elif gtype == "MultiPolygon":
        coords = [[_stitch_ring(ring, arcs) for ring in poly] for poly in geom["arcs"]]
        return {"type": "MultiPolygon", "coordinates": coords}
    return None


def extract_italy_from_world_atlas(topo: dict) -> dict:
    """Return a GeoJSON FeatureCollection with Italy's polygon."""
    transform = topo.get("transform", {})
    scale = transform.get("scale", [1.0, 1.0])
    translate = transform.get("translate", [0.0, 0.0])
    arcs = _decode_topo_arcs(topo.get("arcs", []), scale, translate)

    features = []
    for obj in topo.get("objects", {}).values():
        for geom in obj.get("geometries", []):
            if int(geom.get("id", -1)) == ITALY_COUNTRY_ID:
                decoded = _decode_geometry(geom, arcs)
                if decoded:
                    features.append({
                        "type": "Feature",
                        "id": ITALY_COUNTRY_ID,
                        "properties": {"name": "Italy"},
                        "geometry": decoded,
                    })
    return {"type": "FeatureCollection", "features": features}


# ── helpers ───────────────────────────────────────────────────────────────────
def _download(url: str, label: str) -> bytes:
    print(f"  ↓ Downloading {label} …", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read()
    kb = len(data) / 1024
    print(f"{kb:.0f} KB")
    return data


def _maybe_simplify(src: Path, dst: Path, tolerance: str = "1%") -> None:
    """Run mapshaper simplification if available, else copy as-is."""
    if shutil.which("mapshaper"):
        print(f"  ✂ Simplifying with mapshaper ({tolerance}) …", end=" ", flush=True)
        result = subprocess.run(
            [
                "mapshaper", str(src),
                "-simplify", tolerance, "keep-shapes",
                "-o", "format=geojson", str(dst),
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            kb_before = src.stat().st_size / 1024
            kb_after = dst.stat().st_size / 1024
            print(f"{kb_before:.0f} KB → {kb_after:.0f} KB ({100*kb_after/kb_before:.0f}%)")
            return
        else:
            print(f"FAILED ({result.stderr.strip()[:80]}). Saving full resolution.")
    else:
        print(f"  ℹ  mapshaper not found — saving full resolution. "
              f"Install with: npm install -g mapshaper")
    shutil.copy(src, dst)


def _save_json(data: dict | list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))   # minified
    print(f"  ✓ Saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


# ── main ──────────────────────────────────────────────────────────────────────
def main(overwrite: bool = False) -> None:
    GEO_DIR.mkdir(parents=True, exist_ok=True)

    targets = {
        "italy_outline.geojson": GEO_DIR / "italy_outline.geojson",
        "regions.geojson":       GEO_DIR / "regions.geojson",
        "provinces.geojson":     GEO_DIR / "provinces.geojson",
    }
    if not overwrite and all(p.exists() for p in targets.values()):
        print("All geo files already present. Use --overwrite to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── Italy outline ────────────────────────────────────────────────────
        out_italy = targets["italy_outline.geojson"]
        if overwrite or not out_italy.exists():
            print("\n[1/3] Italy outline (from world-atlas)")
            raw_topo = _download(URLS["world_atlas"], "world-atlas countries-50m.json")
            topo = json.loads(raw_topo)
            italy_geojson = extract_italy_from_world_atlas(topo)
            _save_json(italy_geojson, out_italy)
        else:
            print(f"\n[1/3] Italy outline — already exists, skipping")

        # ── Regions ─────────────────────────────────────────────────────────
        out_regions = targets["regions.geojson"]
        if overwrite or not out_regions.exists():
            print("\n[2/3] Italian regions")
            raw_regions = _download(URLS["regions"], "limits_IT_regions.geojson")
            tmp_regions = tmp_path / "regions_raw.geojson"
            tmp_regions.write_bytes(raw_regions)
            _maybe_simplify(tmp_regions, out_regions)
        else:
            print(f"\n[2/3] Regions — already exists, skipping")

        # ── Provinces ────────────────────────────────────────────────────────
        out_provinces = targets["provinces.geojson"]
        if overwrite or not out_provinces.exists():
            print("\n[3/3] Italian provinces")
            raw_prov = _download(URLS["provinces"], "limits_IT_provinces.geojson")
            tmp_prov = tmp_path / "provinces_raw.geojson"
            tmp_prov.write_bytes(raw_prov)
            _maybe_simplify(tmp_prov, out_provinces)
        else:
            print(f"\n[3/3] Provinces — already exists, skipping")

    # summary
    total_kb = sum(p.stat().st_size for p in targets.values() if p.exists()) / 1024
    print(f"\n✅  Geo bundle ready in {GEO_DIR}")
    print(f"   Total size on disk: {total_kb:.0f} KB")
    print(f"   (vs. ~6 400 KB fetched from network on every first render)")
    print(f"\nNext step: app.py already uses this automatically via _load_geo_bundle().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Italy GeoJSON for D3 dashboards.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
