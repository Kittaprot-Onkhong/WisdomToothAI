"""Side assignment and 'best detection per side' selection."""

from typing import Dict, List, Optional

from .config import LEFT, RIGHT, SIDES
from .detector import Detection


def midpoint(image_width: int) -> int:
    return image_width // 2


def side_for_center(center_x: float, image_width: int) -> str:
    """Centre-x left of the midline -> 'Right' (48); otherwise 'Left' (38)."""
    return RIGHT if center_x < midpoint(image_width) else LEFT


def classify_detections(detections: List[Detection], image_width: int,
                        target_class: str, min_confidence: float) -> Dict[str, Optional[Detection]]:
    """Annotate every raw detection and pick at most ONE tooth per side.

    Every detection is kept (nothing is deleted); the flags/notes record why each one
    was or was not selected. Returns {'Right': Detection|None, 'Left': Detection|None}.
    """
    candidates: Dict[str, List[Detection]] = {side: [] for side in SIDES}

    for det in detections:
        det.side = side_for_center(det.center_x, image_width)
        det.is_target_class = det.class_name.lower() == target_class.lower()
        det.passes_confidence = det.confidence >= min_confidence
        det.selected = False
        if not det.is_target_class:
            det.note = "ignored: not target class"
        elif not det.passes_confidence:
            det.note = "ignored: below confidence threshold"
        else:
            det.note = ""
            candidates[det.side].append(det)

    chosen: Dict[str, Optional[Detection]] = {side: None for side in SIDES}
    for side, items in candidates.items():
        if not items:
            continue
        # highest confidence wins; ties -> earliest in the model's output
        best = max(items, key=lambda d: (d.confidence, -d.index))
        best.selected = True
        best.note = "selected: highest confidence on this side"
        for det in items:
            if det is not best:
                det.note = "rejected: lower confidence than the selected tooth on this side"
        chosen[side] = best
    return chosen


def image_status(chosen: Dict[str, Optional[Detection]]) -> str:
    right, left = chosen[RIGHT] is not None, chosen[LEFT] is not None
    if right and left:
        return "BOTH_SIDES"
    if right:
        return "RIGHT_ONLY"
    if left:
        return "LEFT_ONLY"
    return "NONE_DETECTED"
