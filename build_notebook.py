"""Regenerates WisdomToothAI.ipynb. Run: python build_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = [
md("""
# 🦷 WisdomToothAI — ตรวจจับฟันคุด/ฟันกราม แยกซ้าย-ขวา

- 🔵 กรอบ **น้ำเงิน** = ฝั่งซ้ายของภาพ = **Right / ซี่ 48**
- 🔴 กรอบ **แดง** = ฝั่งขวาของภาพ = **Left / ซี่ 38**
- แสดงได้สูงสุด **1 ซี่ต่อด้าน** (เลือกตัวที่ Confidence สูงสุด) — Raw Detection ทุกกรอบเก็บไว้ใน `raw_detections.csv`

**ก่อนเริ่ม (ทำครั้งเดียว)**
1. Runtime → Change runtime type → **T4 GPU**
2. 🔑 Secrets: เพิ่ม `ROBOFLOW_API_KEY` (เปิด Notebook access) — ถ้า repo เป็น private เพิ่ม `GITHUB_TOKEN` ด้วย
3. กรอกชื่อ GitHub Owner / Repo ในขั้นที่ 1
4. รันทีละช่องตามลำดับ (หรือ Runtime → Run all)
"""),

md("## 🔹 ขั้นที่ 1: ดึงโค้ดจาก GitHub และติดตั้งไลบรารี"),
code('''
#@title ตั้งค่า GitHub Repository { display-mode:"form" }
GITHUB_OWNER = "" #@param {type:"string"}
GITHUB_REPO = "WisdomToothAI" #@param {type:"string"}
GITHUB_BRANCH = "main" #@param {type:"string"}
GITHUB_IMAGE_DIR = "test_images" #@param {type:"string"}

import os, sys, subprocess

if not GITHUB_OWNER.strip():
    raise ValueError("กรอก GITHUB_OWNER (ชื่อบัญชี/องค์กร GitHub ที่เก็บ repo) ก่อน")

try:
    from google.colab import userdata
    GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
except Exception:
    GITHUB_TOKEN = None   # ไม่เป็นไรถ้า repo เป็น public

REPO_DIR = f"/content/{GITHUB_REPO}"
if not os.path.isdir(REPO_DIR):
    auth = f"{GITHUB_TOKEN}@" if GITHUB_TOKEN else ""
    url = f"https://{auth}github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"
    subprocess.run(["git", "clone", "--depth", "1", "-b", GITHUB_BRANCH, url, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("Repo:", REPO_DIR)
print(os.listdir(REPO_DIR))
'''),
code('''
!pip install -q -r requirements.txt
'''),
code('''
import cv2, inference
from importlib.metadata import version
print("OpenCV:", cv2.__version__, "| Inference:", version("inference"))
print("All libraries imported successfully!")
'''),

md("## 🔹 ขั้นที่ 2: โหลด Roboflow API Key จาก Secrets"),
code('''
from google.colab import userdata
import os

try:
    ROBOFLOW_API_KEY = userdata.get("ROBOFLOW_API_KEY")
except Exception:
    ROBOFLOW_API_KEY = None
if not ROBOFLOW_API_KEY:
    raise ValueError("ไม่พบ ROBOFLOW_API_KEY ใน Colab Secrets (🔑 ซ้ายมือ → Add new secret → เปิด Notebook access)")
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
print("Roboflow API Key: Ready")
'''),

md("""
## 🔹 ขั้นที่ 3: ⭐ เลือกโมเดลและตั้งค่า ⭐
เปลี่ยนโมเดลได้ทุกครั้ง แล้วรันขั้นที่ 3 → 8 ใหม่ (ผลแต่ละโมเดลแยกโฟลเดอร์กัน)
"""),
code('''
#@title เลือกโมเดล และตั้งค่า { display-mode:"form" }
from wisdomtooth.config import MODEL_OPTIONS, RunSettings

SELECTED_MODEL = "Roboflow RF-DETR Object Detection (Large)" #@param ["Roboflow RF-DETR Object Detection (Large)", "YOLO26 Object Detection (Large)", "Roboflow 3.0 Object Detection (Large)", "YOLOv11 Object Detection (Large)", "YOLOv12 Object Detection (Large)", "Roboflow RF-DETR XL Object Detection (2X Large)", "YOLO26 Object Detection (X Large)"]
NUM_IMAGES = 100 #@param {type:"integer"}
CONFIDENCE_THRESHOLD = 0.0 #@param {type:"number"}
PREVIEW_COUNT = 5 #@param {type:"integer"}
DRAW_MIDLINE = True #@param {type:"boolean"}

assert SELECTED_MODEL in MODEL_OPTIONS, "ชื่อโมเดลในเมนูไม่ตรงกับ wisdomtooth/config.py"
settings = RunSettings(model_name=SELECTED_MODEL, num_images=NUM_IMAGES,
                       confidence_threshold=CONFIDENCE_THRESHOLD,
                       preview_count=PREVIEW_COUNT, draw_midline=DRAW_MIDLINE)
print("Model      :", settings.model_name)
print("Model ID   :", settings.model_id)
print("Images     :", "ทุกรูป" if NUM_IMAGES == 0 else NUM_IMAGES)
print("Confidence :", settings.confidence_threshold)
'''),

md("## 🔹 ขั้นที่ 4: โหลดโมเดล (Inference Library — ไม่ใช้ API Server)"),
code('''
from wisdomtooth.detector import load_model

print("Loading:", settings.model_id)
model = load_model(settings.model_id, ROBOFLOW_API_KEY)
print("Model loaded successfully!")
'''),

md("## 🔹 ขั้นที่ 5: ดึงรูปทดสอบจาก GitHub โดยตรง"),
code('''
from pathlib import Path
from wisdomtooth.github_images import download_github_images

IMAGE_DIR = Path("/content/test_images_downloaded")
image_paths = download_github_images(
    GITHUB_OWNER, GITHUB_REPO, IMAGE_DIR, path=GITHUB_IMAGE_DIR,
    branch=GITHUB_BRANCH, token=GITHUB_TOKEN, limit=settings.num_images,
)
if settings.num_images and len(image_paths) < settings.num_images:
    print(f"⚠️ ขอ {settings.num_images} รูป แต่ใน GitHub มีเพียง {len(image_paths)} รูป")
'''),

md("## 🔹 ขั้นที่ 6: ⭐ ประมวลผลรูปทั้งหมด ⭐"),
code('''
from wisdomtooth.pipeline import run_pipeline

OUTPUT_ROOT = Path("/content/WisdomToothAI_outputs")
result = run_pipeline(model, settings, image_paths, OUTPUT_ROOT)
'''),

md("## 🔹 ขั้นที่ 7: ดูผลสรุป"),
code('''
import pandas as pd
summary_df = pd.read_csv(result["summary_csv"])
display(summary_df)
display(summary_df["status"].value_counts().rename("จำนวนรูป").to_frame())
'''),
code('''
from google.colab.patches import cv2_imshow
for p in result["preview_images"]:
    print(p.name)
    cv2_imshow(cv2.imread(str(p)))
'''),

md("## 🔹 ขั้นที่ 8: ดาวน์โหลดผลลัพธ์ (ZIP) ลงเครื่อง"),
code('''
from google.colab import files
print("ZIP:", result["zip"])
files.download(str(result["zip"]))
'''),
md("""
**ใน ZIP:** `annotated/` (รูปที่วาดกรอบ) · `raw_detections.csv` (ทุก detection) · `selected_detections.csv` (ที่เลือกแสดง) · `image_summary.csv` (สรุปต่อรูป) · `run_info.json`
"""),
]
nb["cells"] = cells
nb["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU"}
nbf.write(nb, "WisdomToothAI.ipynb")
print("wrote WisdomToothAI.ipynb")
