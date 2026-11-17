"""End-to-end run: images -> model -> filter -> draw -> CSV / ZIP."""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import cv2

from . import __version__
from .config import IMAGE_EXTENSIONS, LEFT, RIGHT, SIDE_TOOTH, RunSettings
from .detector import run_inference
from .export import (RAW_COLUMNS, SELECTED_COLUMNS, SUMMARY_COLUMNS, make_zip,
                     write_csv, write_json)
from .filtering import classify_detections, image_status, midpoint
from .visualize import draw_result


def find_local_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def _lib_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def run_pipeline(model, settings: RunSettings, image_paths: List[Path], output_root: Path,
                 infer_fn: Callable = run_inference, log: Callable = print) -> dict:
    """Process `image_paths` and write all outputs. Returns a dict of paths + totals."""
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise FileNotFoundError("No images to process.")

    output_dir = Path(output_root) / settings.model_short_name
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, selected_rows, summary_rows, preview_paths, times = [], [], [], [], []
    totals = {"processed": 0, "read_errors": 0, "inference_errors": 0,
              "right_detected": 0, "left_detected": 0, "none_detected": 0}
    started = datetime.now()

    log("=" * 65)
    log(f"WisdomToothAI | model: {settings.model_id} | images: {len(image_paths)}")
    log("=" * 65)

    for number, path in enumerate(image_paths, start=1):
        log(f"[{number}/{len(image_paths)}] {path.name}")
        summary = {"image": path.name, "width": "", "height": "", "midpoint_x": "",
                   "raw_detection_count": 0, "target_class_count": 0,
                   "right_48_detected": 0, "right_48_confidence": "",
                   "left_38_detected": 0, "left_38_confidence": "",
                   "selected_count": 0, "status": "", "inference_seconds": "",
                   "annotated_file": ""}

        image = cv2.imread(str(path))
        if image is None:
            log("  [ERROR] cannot read image")
            summary["status"] = "READ_ERROR"
            totals["read_errors"] += 1
            summary_rows.append(summary)
            continue

        height, width = image.shape[:2]
        summary.update(width=width, height=height, midpoint_x=midpoint(width))

        try:
            t0 = time.perf_counter()
            detections = infer_fn(model, image, settings.confidence_threshold)
            elapsed = time.perf_counter() - t0
        except Exception as error:  # keep going; record the failure
            log(f"  [ERROR] inference failed: {error}")
            summary["status"] = "INFERENCE_ERROR"
            totals["inference_errors"] += 1
            summary_rows.append(summary)
            continue
        times.append(elapsed)

        chosen = classify_detections(detections, width, settings.target_class,
                                     settings.confidence_threshold)
        status = image_status(chosen)

        # ---- raw detections: EVERYTHING the model returned ----
        for det in detections:
            raw_rows.append({
                "image": path.name, "image_width": width, "image_height": height,
                "midpoint_x": midpoint(width), "detection_index": det.index,
                "class": det.class_name, "confidence": det.confidence,
                "center_x": det.center_x, "center_y": det.center_y,
                "box_width": det.width, "box_height": det.height,
                "bbox_x1": det.x1, "bbox_y1": det.y1, "bbox_x2": det.x2, "bbox_y2": det.y2,
                "side": det.side, "side_class": f"Tooth_{SIDE_TOOTH[det.side]}",
                "is_target_class": det.is_target_class,
                "passes_confidence": det.passes_confidence,
                "selected": det.selected, "note": det.note,
            })

        for side, det in chosen.items():
            if det is None:
                continue
            selected_rows.append({
                "image": path.name, "side": side, "side_class": f"Tooth_{SIDE_TOOTH[side]}",
                "class": det.class_name, "confidence": det.confidence,
                "center_x": det.center_x, "center_y": det.center_y,
                "box_width": det.width, "box_height": det.height,
                "bbox_x1": det.x1, "bbox_y1": det.y1, "bbox_x2": det.x2, "bbox_y2": det.y2,
            })

        annotated = draw_result(image, chosen, settings.draw_midline)
        out_path = annotated_dir / f"{path.stem}_result.jpg"
        cv2.imwrite(str(out_path), annotated)
        if len(preview_paths) < settings.preview_count:
            preview_paths.append(out_path)

        right, left = chosen[RIGHT], chosen[LEFT]
        summary.update(
            raw_detection_count=len(detections),
            target_class_count=sum(d.is_target_class for d in detections),
            right_48_detected=int(right is not None),
            right_48_confidence=round(right.confidence, 6) if right else "",
            left_38_detected=int(left is not None),
            left_38_confidence=round(left.confidence, 6) if left else "",
            selected_count=int(right is not None) + int(left is not None),
            status=status, inference_seconds=round(elapsed, 4), annotated_file=out_path.name,
        )
        summary_rows.append(summary)

        totals["processed"] += 1
        totals["right_detected"] += int(right is not None)
        totals["left_detected"] += int(left is not None)
        totals["none_detected"] += int(status == "NONE_DETECTED")
        log(f"  {width}x{height} | raw={len(detections)} | Right/48="
            f"{'%.2f' % right.confidence if right else '-'} | Left/38="
            f"{'%.2f' % left.confidence if left else '-'} | {status} | {elapsed:.2f}s")

    # ---------------- write outputs ----------------
    paths = {
        "output_dir": output_dir,
        "annotated_dir": annotated_dir,
        "raw_csv": write_csv(output_dir / "raw_detections.csv", RAW_COLUMNS, raw_rows),
        "selected_csv": write_csv(output_dir / "selected_detections.csv", SELECTED_COLUMNS, selected_rows),
        "summary_csv": write_csv(output_dir / "image_summary.csv", SUMMARY_COLUMNS, summary_rows),
    }
    average = sum(times) / len(times) if times else 0.0
    paths["run_info"] = write_json(output_dir / "run_info.json", {
        "project_version": __version__,
        "run_started": started.isoformat(timespec="seconds"),
        "model_name": settings.model_name, "model_id": settings.model_id,
        "target_class": settings.target_class,
        "confidence_threshold": settings.confidence_threshold,
        "num_images_requested": settings.num_images, "images_tested": len(image_paths),
        "rule": "max 1 tooth per side, highest confidence; image-left = Right/48 (blue), "
                "image-right = Left/38 (red)",
        "totals": totals, "average_inference_seconds": round(average, 4),
        "library_versions": {"inference": _lib_version("inference"),
                             "opencv": cv2.__version__},
    })
    stamp = started.strftime("%Y%m%d_%H%M%S")
    paths["zip"] = make_zip(output_dir, Path(output_root) / f"WisdomToothAI_{settings.model_short_name}_{stamp}.zip")
    paths["preview_images"] = preview_paths
    paths["totals"] = totals
    paths["average_inference_seconds"] = average

    log("\n" + "=" * 65)
    log("PROCESSING COMPLETE")
    log(f"Tested {len(image_paths)} | processed {totals['processed']} | "
        f"read errors {totals['read_errors']} | inference errors {totals['inference_errors']}")
    log(f"Right/48 found: {totals['right_detected']} | Left/38 found: {totals['left_detected']} | "
        f"none: {totals['none_detected']} | avg {average:.3f}s/image")
    log(f"ZIP: {paths['zip']}")
    log("=" * 65)
    return paths
