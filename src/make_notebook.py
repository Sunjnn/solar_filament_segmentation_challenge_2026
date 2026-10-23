#!/usr/bin/env python3
"""Convert percent-format .py files to .ipynb notebooks.

Usage:
    python src/make_notebook.py                  # Convert all src/*.py → kaggle_*/**.ipynb
    python src/make_notebook.py src/foo.py       # Convert a single file
"""

import json
import re
import sys
from pathlib import Path

import nbformat as nbf

# Mapping: source .py → target .ipynb
NOTEBOOK_MAP = {
    "filament_baseline.py": "kaggle_baseline/filament_baseline.ipynb",
    "filament_eda.py": "kaggle_baseline/filament_eda.ipynb",
    "exp_domain.py": "kaggle_baseline/exp_domain.ipynb",
    "smoke_check.py": "kaggle_baseline/smoke_check.ipynb",
    "filament_yolo_refiner.py": "kaggle_yolo_refiner/filament_yolo_refiner.ipynb",
}

PERCENT_SEP = re.compile(r"^# %%\s*(.*)$")


def parse_percent_file(path: Path) -> list[dict]:
    """Parse a percent-format .py file into notebook cells."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    cells: list[dict] = []
    current_type = "code"
    current_lines: list[str] = []

    for line in lines:
        m = PERCENT_SEP.match(line)
        if m:
            # Save previous cell
            if current_lines:
                cells.append({"cell_type": current_type, "source": "\n".join(current_lines)})
                current_lines = []

            marker = m.group(1).strip().lower()
            if marker.startswith("md ") or marker.startswith("markdown "):
                current_type = "markdown"
            else:
                current_type = "code"
        else:
            current_lines.append(line)

    # Save last cell
    if current_lines:
        cells.append({"cell_type": current_type, "source": "\n".join(current_lines)})

    return cells


def build_notebook(cells: list[dict]) -> nbf.NotebookNode:
    """Build an nbformat notebook from cell dicts."""
    nb_cells = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(cell["source"]))
        else:
            nb_cells.append(nbf.v4.new_code_cell(cell["source"]))

    nb = nbf.v4.new_notebook(cells=nb_cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0",
        },
    }
    return nb


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"

    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(src_dir.glob("*.py"))

    for src_path in targets:
        if src_path.name == "make_notebook.py":
            continue

        dest_rel = NOTEBOOK_MAP.get(src_path.name)
        if dest_rel is None:
            print(f"⚠  Skipping {src_path.name} — no mapping in NOTEBOOK_MAP")
            continue

        dest_path = project_root / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"🔄 {src_path.name} → {dest_rel}")
        cells = parse_percent_file(src_path)
        nb = build_notebook(cells)
        nbf.write(nb, dest_path)
        print(f"✅ Wrote {len(cells)} cells to {dest_rel}")

    print("Done.")


if __name__ == "__main__":
    main()