"""Load a Roboflow model with the Inference *library* (no API server) and run it."""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class Detection:
    """One raw prediction, normalised. Filled in further by filtering.py."""
    index: int                 # position in the model's raw output (0-based)
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float
    # --- set by filtering.classify_detections ---
    side: str = ""             # "Right" / "Left"
    is_target_class: bool = False
    passes_confidence: bool = False
    selected: bool = False
    note: str = ""

    @property
    def x1(self) -> float:
        return self.center_x - self.width / 2

    @property
    def y1(self) -> float:
        return self.center_y - self.height / 2

    @property
    def x2(self) -> float:
        return self.center_x + self.width / 2

    @property
    def y2(self) -> float:
        return self.center_y + self.height / 2


def load_model(model_id: str, api_key: str):
    """Load the model locally via inference.get_model() (weights are downloaded once)."""
    from inference import get_model  # imported lazily so tests run without the library

    if not api_key:
        raise ValueError("Roboflow API key is empty.")
    return get_model(model_id=model_id, api_key=api_key)


def _field(prediction: Any, name: str, default=None):
    if isinstance(prediction, dict):
        return prediction.get(name, default)
    return getattr(prediction, name, default)


def to_detection(index: int, prediction: Any) -> Detection:
    return Detection(
        index=index,
        class_name=str(_field(prediction, "class_name", _field(prediction, "class", "unknown"))),
        confidence=float(_field(prediction, "confidence", 0.0)),
        center_x=float(_field(prediction, "x", 0.0)),
        center_y=float(_field(prediction, "y", 0.0)),
        width=float(_field(prediction, "width", 0.0)),
        height=float(_field(prediction, "height", 0.0)),
    )


def run_inference(model, image, confidence: float = 0.0) -> List[Detection]:
    """Run the model on the full BGR image (no crop / resize) and return ALL detections.

    `confidence` is passed to the model so that low-confidence boxes are kept in the raw
    output (the library default of 0.4 would otherwise hide them).
    """
    try:
        result = model.infer(image, confidence=confidence)
    except TypeError:  # a model class that does not accept the keyword
        result = model.infer(image)
    if isinstance(result, (list, tuple)):
        result = result[0]
    predictions = _field(result, "predictions", []) or []
    return [to_detection(i, p) for i, p in enumerate(predictions)]
