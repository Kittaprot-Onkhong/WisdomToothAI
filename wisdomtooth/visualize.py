"""Draw the result: blue box = Right/48, red box = Left/38, text with a dark outline."""

from typing import Dict, Optional

import cv2

from .config import (MIDLINE_COLOR, OUTLINE_COLOR, RIGHT, SIDES, SIDE_COLOR,
                     SIDE_TOOTH, TEXT_SCALE)
from .detector import Detection
from .filtering import midpoint

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_outlined_text(image, text, origin, color, scale, thickness):
    """Coloured text with a dark border so it stays readable on any background.

    The border is made by drawing the text in the outline colour shifted in 8 directions
    (instead of relying on putText's thickness, which differs between OpenCV versions).
    """
    x, y = origin
    r = max(1, int(round(scale * 2)))            # border width in pixels
    for dx in (-r, 0, r):
        for dy in (-r, 0, r):
            if dx or dy:
                cv2.putText(image, text, (x + dx, y + dy), FONT, scale, OUTLINE_COLOR,
                            thickness, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _clamp(value, low, high):
    return max(low, min(high, value))


def draw_result(image, chosen: Dict[str, Optional[Detection]], draw_midline: bool = True):
    """Return a copy of `image` with the selected detections (max 1 per side) drawn."""
    out = image.copy()
    height, width = out.shape[:2]

    base = width / 1500                              # scales with image size
    scale = max(0.4, base * TEXT_SCALE)              # text size (set TEXT_SCALE in config.py)
    thickness = max(1, int(round(scale * 2)))
    box_thickness = max(2, int(round(base * 3)))     # box line keeps its normal width

    if draw_midline:
        cv2.line(out, (midpoint(width), 0), (midpoint(width), height - 1), MIDLINE_COLOR, 2)

    for side in SIDES:
        det = chosen.get(side)
        if det is None:
            continue
        color = SIDE_COLOR[side]
        x1 = int(_clamp(round(det.x1), 0, width - 1))
        y1 = int(_clamp(round(det.y1), 0, height - 1))
        x2 = int(_clamp(round(det.x2), 0, width - 1))
        y2 = int(_clamp(round(det.y2), 0, height - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)

        label = f"{side} {SIDE_TOOTH[side]} | {det.class_name} {det.confidence:.2f}"
        (_, text_h), baseline = cv2.getTextSize(label, FONT, scale, thickness)
        label_y = y1 - 8 if y1 - 8 - text_h > 0 else y1 + text_h + 8   # keep text inside image
        label_y = int(_clamp(label_y, text_h + 2, height - baseline - 2))
        label_x = int(_clamp(x1, 0, max(0, width - 10)))
        put_outlined_text(out, label, (label_x, label_y), color, scale, thickness)

    # counts / "not detected" notes, top-left corner
    line_gap = int(34 * scale) + 8
    y = line_gap
    for side in SIDES:
        count = 1 if chosen.get(side) is not None else 0
        put_outlined_text(out, f"{side} / {SIDE_TOOTH[side]}: {count}", (15, y),
                          SIDE_COLOR[side], scale * 0.9, thickness)
        y += line_gap
    for side in SIDES:
        if chosen.get(side) is None:
            put_outlined_text(out, f"No tooth detected on {side} / {SIDE_TOOTH[side]}", (15, y),
                              SIDE_COLOR[side], scale * 0.65, max(1, thickness - 1))
            y += int(line_gap * 0.8)
    return out