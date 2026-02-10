#!/usr/bin/env python3
"""
List all content under a main folder recursively and write to a text file with:
1) A TREE (indented) view
2) A FLAT ([DIR]/[FILE]) view

Optional: filter files by extension(s).
Also logs file sizes for files.

Examples:
  python list_tree_to_txt.py "D:/my_project" -o tree.txt
  python list_tree_to_txt.py "D:/my_project" -o tree.txt --ext .py .md
  python list_tree_to_txt.py "D:/data" -o data_tree.txt --ext parquet --no-hidden
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional, Set, List, Tuple


def normalize_exts(exts: Optional[Iterable[str]]) -> Optional[Set[str]]:
    """Normalize extensions to a set like {'.py', '.md'} (lowercase). Accepts 'py' or '.py'."""
    if not exts:
        return None
    out = set()
    for e in exts:
        e = (e or "").strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.add(e)
    return out or None


def should_include_file(path: Path, exts: Optional[Set[str]]) -> bool:
    if exts is None:
        return True
    return path.suffix.lower() in exts


def human_bytes(num: int) -> str:
    """Human-readable bytes (binary)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    n = float(num)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(n)} {u}"
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{num} B"


def safe_stat_size(p: Path) -> int:
    """Get file size; return -1 if inaccessible."""
    try:
        return p.stat().st_size
    except Exception:
        return -1


def build_views(
    root: Path,
    exts: Optional[Set[str]] = None,
    include_hidden: bool = True,
    sort_entries: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      tree_lines: indented tree view
      flat_lines: flat list with [DIR]/[FILE]
    """
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    root = root.resolve()

    # TREE VIEW (depth-indented by relative path parts)
    tree_lines: List[str] = []
    flat_lines: List[str] = []

    # We'll walk directories and print each directory once, then its files, all at the right indent.
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)

        # Optionally ignore hidden items (leading dot)
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]

        if sort_entries:
            dirnames.sort(key=str.lower)
            filenames.sort(key=str.lower)

        rel_dir = dirpath_p.relative_to(root)
        depth = 0 if str(rel_dir) == "." else len(rel_dir.parts)

        # TREE: print directory line
        dir_label = "." if str(rel_dir) == "." else rel_dir.name
        tree_lines.append(f"{'│   ' * max(depth - 1, 0)}{'├── ' if depth else ''}{dir_label}/  [DIR]")

        # FLAT: print directory line
        flat_lines.append(f"[DIR]  {rel_dir.as_posix() if str(rel_dir) != '.' else '.'}")

        # TREE: print files within this dir (filtered)
        for fn in filenames:
            fp = dirpath_p / fn
            if not should_include_file(fp, exts):
                continue
            size_b = safe_stat_size(fp)
            size_str = "N/A" if size_b < 0 else f"{size_b} B ({human_bytes(size_b)})"

            # indent files one level deeper than their directory
            file_prefix = "│   " * depth + "├── "
            tree_lines.append(f"{file_prefix}{fn}  [FILE]  {size_str}")

            # FLAT: relative path + sizes
            rel_file = fp.relative_to(root).as_posix()
            flat_lines.append(f"[FILE] {rel_file}  |  {size_str}")

    return tree_lines, flat_lines


def write_log(
    output_path: Path,
    root: Path,
    exts: Optional[Set[str]],
    tree_lines: List[str],
    flat_lines: List[str],
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ext_str = "ALL" if exts is None else ", ".join(sorted(exts))

    header = [
        "Folder listing log",
        f"Timestamp: {ts}",
        f"Root: {root.resolve()}",
        f"File filter extensions: {ext_str}",
        "-" * 80,
        "",
        "TREE VIEW",
        "-" * 80,
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n")
        f.write("\n".join(tree_lines))
        f.write("\n\n")
        f.write("-" * 80)
        f.write("\nFLAT VIEW ([DIR]/[FILE])\n")
        f.write("-" * 80)
        f.write("\n")
        f.write("\n".join(flat_lines))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export folder contents (tree + flat) to a text file, optionally filtering by file extension."
    )
    parser.add_argument("root_folder", type=str, help="Main folder to scan recursively.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="folder_listing.txt",
        help="Output text file path (default: folder_listing.txt).",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include (e.g. --ext .py .md json). If omitted: include all files.",
    )
    parser.add_argument(
        "--no-hidden",
        action="store_true",
        help="Ignore hidden files/folders (starting with '.').",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort directory and file names.",
    )

    args = parser.parse_args()

    root = Path(args.root_folder)
    output = Path(args.output)
    exts = normalize_exts(args.ext)

    tree_lines, flat_lines = build_views(
        root=root,
        exts=exts,
        include_hidden=not args.no_hidden,
        sort_entries=not args.no_sort,
    )

    write_log(output, root, exts, tree_lines, flat_lines)
    print(f"Done. Wrote log to: {output.resolve()}")


if __name__ == "__main__":
    main()
