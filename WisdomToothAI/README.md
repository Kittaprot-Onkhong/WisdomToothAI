# 🦷 WisdomToothAI

ตรวจจับฟันคุด/ฟันกราม (class `molar`) ในภาพ แล้วแยกซ้าย-ขวา รันบน **Google Colab**
โดยโหลดโมเดล Roboflow ด้วย **Inference Library** (`inference.get_model()`) — ไม่ใช้ API Server

## กติกาของระบบ
| ข้อ | รายละเอียด |
|---|---|
| ด้าน | กึ่งกลางภาพ = `width // 2` · กรอบที่ center-x อยู่ **ซ้ายของภาพ = Right / ซี่ 48 (🔵 น้ำเงิน)** · **ขวาของภาพ = Left / ซี่ 38 (🔴 แดง)** |
| จำกัดจำนวน | แสดงได้ **สูงสุด 1 ซี่ต่อด้าน** — เลือกกรอบที่ Confidence สูงสุด (เท่ากัน = กรอบที่โมเดลให้ก่อน) |
| Raw detection | ทุกกรอบที่โมเดลตอบ (รวมคลาสอื่น/กรอบที่ถูกตัดทิ้ง) เก็บใน `raw_detections.csv` พร้อมคอลัมน์ `selected` และ `note` บอกเหตุผล |
| ภาพผลลัพธ์ | กรอบสี + ข้อความที่มี **ขอบสีดำ** + เส้นเหลืองกลางภาพ (ปิดได้ด้วย `DRAW_MIDLINE`) |
| โมเดล | เลือกจากเมนู 7 โมเดลใน `wisdomtooth/config.py` (`MODEL_OPTIONS`) |

## โครงสร้าง Repository
```
WisdomToothAI/
├── WisdomToothAI.ipynb        # Notebook สำหรับ Colab (เปิดจาก GitHub ได้โดยตรง)
├── requirements.txt
├── build_notebook.py          # สร้าง .ipynb ใหม่ (python build_notebook.py)
├── wisdomtooth/
│   ├── config.py              # MODEL_OPTIONS, สี, กติกาซ้าย/ขวา, RunSettings
│   ├── github_images.py       # ดึงรูปจาก GitHub (Contents API)
│   ├── detector.py            # get_model() + รัน inference + Detection
│   ├── filtering.py           # แยกด้าน + เลือก 1 ซี่/ด้าน
│   ├── visualize.py           # วาดกรอบ/ข้อความมีขอบ
│   ├── export.py              # CSV / JSON / ZIP
│   └── pipeline.py            # ประกอบทุกอย่างเข้าด้วยกัน
├── tests/test_wisdomtooth.py  # เทสต์ออฟไลน์ (ไม่ต้องมี API key)
└── test_images/               # ← อัปโหลดรูปทดสอบ 100 รูปที่นี่
```

## วิธีใช้
1. สร้าง GitHub repo แล้ว push ไฟล์ทั้งหมดนี้ขึ้นไป จากนั้นอัปโหลดรูปทดสอบไปที่ `test_images/`
2. เปิด `WisdomToothAI.ipynb` ใน Colab (File → Open notebook → GitHub) แล้วเลือก **T4 GPU**
3. ใน 🔑 Secrets เพิ่ม `ROBOFLOW_API_KEY` (เปิด Notebook access) — ถ้า repo เป็น **private** เพิ่ม `GITHUB_TOKEN` ด้วย
4. ขั้นที่ 1: กรอก `GITHUB_OWNER` (และ `GITHUB_REPO` / `GITHUB_BRANCH` ถ้าต่างจากค่าเริ่มต้น) แล้วรันทีละช่อง
5. ขั้นที่ 3 เลือกโมเดล (`NUM_IMAGES` ค่าเริ่มต้น 100, `0` = ทุกรูป) · ขั้นที่ 8 ดาวน์โหลด ZIP

เปลี่ยนโมเดล → รันขั้นที่ 3 → 8 ใหม่ (ไม่ต้องติดตั้งซ้ำ)

## ผลลัพธ์ (ใน ZIP)
| ไฟล์ | เนื้อหา |
|---|---|
| `annotated/*_result.jpg` | รูปที่วาดกรอบแล้ว |
| `image_summary.csv` | 1 รูป = 1 แถว: จำนวน raw, conf ของ 48 / 38, `status` (`BOTH_SIDES`, `RIGHT_ONLY`, `LEFT_ONLY`, `NONE_DETECTED`, `READ_ERROR`, `INFERENCE_ERROR`), เวลา inference |
| `raw_detections.csv` | ทุก detection ที่โมเดลคืนมา + ด้าน + `selected` + `note` |
| `selected_detections.csv` | เฉพาะกรอบที่เลือกแสดง (≤ 1 ต่อด้านต่อรูป) |
| `run_info.json` | โมเดล, พารามิเตอร์, เวอร์ชันไลบรารี, ยอดรวม |

## หมายเหตุ
- `CONFIDENCE_THRESHOLD` (ค่าเริ่มต้น `0.0` ตามโน้ตบุ๊กเดิม) ถูกส่งให้โมเดลโดยตรงด้วย เพื่อให้ Raw ไม่ถูกตัดที่ 0.4 ซึ่งเป็นค่าเริ่มต้นของไลบรารี
- ข้อความบนภาพเป็นภาษาอังกฤษ เพราะ OpenCV วาดตัวอักษรไทยไม่ได้
- ทดสอบโค้ด: `python -m unittest discover -s tests -v`
