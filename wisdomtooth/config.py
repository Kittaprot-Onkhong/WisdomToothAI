"""Central configuration. Everything project-specific lives here."""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model options (copied verbatim from the original WisdomToothAI notebook).
# Key   = name shown in the Colab dropdown
# Value = Roboflow model id passed to inference.get_model()
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "Roboflow RF-DETR Object Detection (Large)": "krittaporns-workspace/molar-detect-2-1-rfdetr-large-t1",
    "YOLO26 Object Detection (Large)": "krittaporns-workspace/molar-detect-2-1-yolo26l-t2",
    "Roboflow 3.0 Object Detection (Large)": "krittaporns-workspace/molar-detect-2-1-yolov8l-t4",
    "YOLOv11 Object Detection (Large)": "krittaporns-workspace/molar-detect-2-1-yolo11l-t3",
    "YOLOv12 Object Detection (Large)": "krittaporns-workspace/molar-detect-2-1-yolo12l-t6",
    "Roboflow RF-DETR XL Object Detection (2X Large)": "krittaporns-workspace/molar-detect-2-2-rfdetr-2xlarge-t1",
    "YOLO26 Object Detection (X Large)": "krittaporns-workspace/molar-detect-2-2-yolo26x-t2",
}
DEFAULT_MODEL_NAME = "Roboflow RF-DETR Object Detection (Large)"

# Class name the models output for the tooth (from the original notebook).
TARGET_CLASS = "molar"

# ---------------------------------------------------------------------------
# Side rules (from the original notebook).
# The image is split at x = width // 2:
#   centre-x LEFT of the midline  -> "Right" tooth 48 -> BLUE
#   centre-x RIGHT of the midline -> "Left"  tooth 38 -> RED
# Colours are OpenCV BGR.
# ---------------------------------------------------------------------------
RIGHT = "Right"
LEFT = "Left"
SIDES = (RIGHT, LEFT)
SIDE_TOOTH = {RIGHT: "48", LEFT: "38"}
SIDE_COLOR = {RIGHT: (255, 0, 0), LEFT: (0, 0, 255)}  # BGR: blue / red
OUTLINE_COLOR = (0, 0, 0)                              # dark text border
MIDLINE_COLOR = (0, 255, 255)                          # yellow reference line
TEXT_SCALE = 0.7

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# GitHub location of the test images. Fill these in once the repo exists
# (or type them into the Colab form). Empty on purpose — never guessed.
# ---------------------------------------------------------------------------
GITHUB_OWNER = ""
GITHUB_REPO = ""
GITHUB_BRANCH = "main"
GITHUB_IMAGE_DIR = "test_images"


@dataclass
class RunSettings:
    model_name: str = DEFAULT_MODEL_NAME
    num_images: int = 100            # 0 = every image found
    confidence_threshold: float = 0.0
    preview_count: int = 5
    target_class: str = TARGET_CLASS
    draw_midline: bool = True

    @property
    def model_id(self) -> str:
        if self.model_name not in MODEL_OPTIONS:
            raise KeyError(
                f"Unknown model '{self.model_name}'. Choose one of: {list(MODEL_OPTIONS)}"
            )
        return MODEL_OPTIONS[self.model_name]

    @property
    def model_short_name(self) -> str:
        return self.model_id.split("/")[-1]
