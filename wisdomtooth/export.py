"""CSV, run-info and ZIP writers."""

import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, List

RAW_COLUMNS = [
    "image", "image_width", "image_height", "midpoint_x",
    "detection_index", "class", "confidence",
    "center_x", "center_y", "box_width", "box_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "side", "side_class", "is_target_class", "passes_confidence", "selected", "note",
]

SELECTED_COLUMNS = [
    "image", "side", "side_class", "class", "confidence",
    "center_x", "center_y", "box_width", "box_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]

SUMMARY_COLUMNS = [
    "image", "width", "height", "midpoint_x",
    "raw_detection_count", "target_class_count",
    "right_48_detected", "right_48_confidence",
    "left_38_detected", "left_38_confidence",
    "selected_count", "status", "inference_seconds", "annotated_file",
]


def write_csv(path: Path, columns: List[str], rows: List[Dict]) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:  # utf-8-sig: opens cleanly in Excel
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, data: Dict) -> Path:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return Path(path)


def make_zip(source_dir: Path, zip_path: Path) -> Path:
    """Zip everything under source_dir (annotated images, CSVs, run_info.json)."""
    source_dir, zip_path = Path(source_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(source_dir.rglob("*")):
            if file.is_file():
                archive.write(file, Path(source_dir.name) / file.relative_to(source_dir))
    return zip_path
