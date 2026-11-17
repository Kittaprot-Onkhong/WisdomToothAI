"""Offline tests (no Roboflow / GitHub access needed). Run: python -m unittest discover -s tests -v"""
import ast
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wisdomtooth import github_images
from wisdomtooth.config import MODEL_OPTIONS, RunSettings
from wisdomtooth.detector import Detection, run_inference
from wisdomtooth.filtering import classify_detections, side_for_center
from wisdomtooth.pipeline import find_local_images, run_pipeline


def pred(cls, conf, x, y=100, w=40, h=60):
    return SimpleNamespace(class_name=cls, confidence=conf, x=x, y=y, width=w, height=h)


class FakeModel:
    """Returns canned predictions keyed by image width so each image behaves differently."""
    def __init__(self, table):
        self.table = table

    def infer(self, image, confidence=0.4):
        width = image.shape[1]
        if width not in self.table:
            raise RuntimeError("boom")
        preds = [p for p in self.table[width] if p.confidence >= confidence]
        return [SimpleNamespace(predictions=preds)]


class SideAndSelection(unittest.TestCase):
    def test_side_rule(self):
        self.assertEqual(side_for_center(99, 200), "Right")    # left half of image -> Right/48
        self.assertEqual(side_for_center(100, 200), "Left")    # on/after midline -> Left/38
        self.assertEqual(side_for_center(150, 200), "Left")

    def test_max_one_per_side_highest_confidence(self):
        dets = [
            Detection(0, "molar", 0.60, 50, 50, 10, 10),
            Detection(1, "molar", 0.90, 60, 50, 10, 10),    # best Right
            Detection(2, "molar", 0.30, 150, 50, 10, 10),   # only Left
            Detection(3, "other", 0.99, 160, 50, 10, 10),   # wrong class, never selected
        ]
        chosen = classify_detections(dets, 200, "molar", 0.0)
        self.assertIs(chosen["Right"], dets[1])
        self.assertIs(chosen["Left"], dets[2])
        self.assertEqual(sum(d.selected for d in dets), 2)
        self.assertIn("rejected", dets[0].note)
        self.assertIn("not target", dets[3].note)

    def test_threshold_removes_candidates(self):
        dets = [Detection(0, "molar", 0.40, 50, 50, 10, 10)]
        chosen = classify_detections(dets, 200, "molar", 0.5)
        self.assertIsNone(chosen["Right"])
        self.assertFalse(dets[0].selected)

    def test_run_inference_fallback_without_confidence_kwarg(self):
        class Old:
            def infer(self, image):
                return [SimpleNamespace(predictions=[pred("molar", 0.5, 10)])]
        self.assertEqual(len(run_inference(Old(), np.zeros((10, 10, 3), np.uint8), 0.0)), 1)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        img_dir = self.tmp / "imgs"
        img_dir.mkdir()
        # width encodes the scenario for FakeModel
        cv2.imwrite(str(img_dir / "a_both.jpg"), np.full((300, 400, 3), 128, np.uint8))
        cv2.imwrite(str(img_dir / "b_right.jpg"), np.full((300, 410, 3), 128, np.uint8))
        cv2.imwrite(str(img_dir / "c_none.jpg"), np.full((300, 420, 3), 128, np.uint8))
        cv2.imwrite(str(img_dir / "d_err.jpg"), np.full((300, 430, 3), 128, np.uint8))
        (img_dir / "e_bad.jpg").write_bytes(b"not an image")
        (img_dir / "notes.txt").write_text("ignored")
        self.img_dir = img_dir
        self.model = FakeModel({
            400: [pred("molar", 0.80, 100), pred("molar", 0.95, 110), pred("molar", 0.70, 300),
                  pred("molar", 0.55, 310), pred("tooth", 0.99, 320)],
            410: [pred("molar", 0.88, 90)],
            420: [],
        })

    def test_full_run(self):
        paths = find_local_images(self.img_dir)
        self.assertEqual(len(paths), 5)  # .txt skipped
        out = self.tmp / "out"
        logs = []
        res = run_pipeline(self.model, RunSettings(), paths, out, log=logs.append)

        # raw CSV keeps ALL 6 detections incl. rejected / wrong class
        raw = pd.read_csv(res["raw_csv"], encoding="utf-8-sig")
        self.assertEqual(len(raw), 6)  # 5 (both-sides image) + 1 (right-only) + 0 (none)
        self.assertEqual(int(raw["selected"].sum()), 3)  # both-sides image: 2, right-only: 1

        # at most 1 selected per side per image
        sel = pd.read_csv(res["selected_csv"], encoding="utf-8-sig")
        self.assertTrue((sel.groupby(["image", "side"]).size() == 1).all())
        a = sel[sel.image == "a_both.jpg"].set_index("side")
        self.assertAlmostEqual(a.loc["Right", "confidence"], 0.95)
        self.assertAlmostEqual(a.loc["Left", "confidence"], 0.70)

        summ = pd.read_csv(res["summary_csv"], encoding="utf-8-sig").set_index("image")
        self.assertEqual(summ.loc["a_both.jpg", "status"], "BOTH_SIDES")
        self.assertEqual(summ.loc["b_right.jpg", "status"], "RIGHT_ONLY")
        self.assertEqual(summ.loc["c_none.jpg", "status"], "NONE_DETECTED")
        self.assertEqual(summ.loc["d_err.jpg", "status"], "INFERENCE_ERROR")
        self.assertEqual(summ.loc["e_bad.jpg", "status"], "READ_ERROR")
        self.assertEqual(summ.loc["a_both.jpg", "raw_detection_count"], 5)

        # annotated images: blue box on the Right/48 side, red on Left/38 (BGR)
        img = cv2.imread(str(res["annotated_dir"] / "a_both_result.jpg"))
        blue = img[:, :200].reshape(-1, 3)
        red = img[:, 200:].reshape(-1, 3)
        is_blue = (blue[:, 0] > 200) & (blue[:, 1] < 60) & (blue[:, 2] < 60)
        is_red = (red[:, 2] > 200) & (red[:, 0] < 60) & (red[:, 1] < 60)
        self.assertGreater(is_blue.sum(), 50)
        self.assertGreater(is_red.sum(), 50)
        # dark text outline exists (checked on the uncompressed drawing; JPEG softens it)
        from wisdomtooth.filtering import classify_detections as _cls
        from wisdomtooth.visualize import draw_result
        base = np.full((300, 400, 3), 128, np.uint8)
        dets = [Detection(0, "molar", 0.9, 100, 150, 40, 60)]
        drawn = draw_result(base, _cls(dets, 400, "molar", 0.0), draw_midline=False)
        self.assertGreater(int((drawn.sum(axis=2) == 0).sum()), 50)
        self.assertEqual(len(list(res["annotated_dir"].glob("*.jpg"))), 3)

        # ZIP contains everything
        with zipfile.ZipFile(res["zip"]) as z:
            names = " ".join(z.namelist())
        for needle in ("raw_detections.csv", "selected_detections.csv", "image_summary.csv",
                       "run_info.json", "a_both_result.jpg"):
            self.assertIn(needle, names)
        info = json.loads(res["run_info"].read_text(encoding="utf-8"))
        self.assertEqual(info["model_id"], MODEL_OPTIONS[RunSettings().model_name])

    def test_every_model_option_resolves(self):
        for name, mid in MODEL_OPTIONS.items():
            s = RunSettings(model_name=name)
            self.assertEqual(s.model_id, mid)
            self.assertTrue(s.model_short_name)

    def test_confidence_threshold_filters(self):
        res = run_pipeline(self.model, RunSettings(confidence_threshold=0.9),
                           find_local_images(self.img_dir)[:1], self.tmp / "o2", log=lambda *_: None)
        sel = pd.read_csv(res["selected_csv"], encoding="utf-8-sig")
        self.assertEqual(len(sel), 1)               # only 0.95 survives
        raw = pd.read_csv(res["raw_csv"], encoding="utf-8-sig")
        self.assertEqual(len(raw), 2)               # model itself cut at 0.9 via confidence arg


class GitHubDownload(unittest.TestCase):
    def test_download(self):
        listing = [
            {"type": "file", "name": "b.png", "size": 3, "download_url": "https://raw/b.png"},
            {"type": "file", "name": "a.jpg", "size": 3, "download_url": "https://raw/a.jpg"},
            {"type": "file", "name": "readme.md", "size": 1, "download_url": "https://raw/readme.md"},
            {"type": "dir", "name": "sub", "size": 0, "download_url": None},
        ]

        class Resp:
            def __init__(self, data): self.data = data
            def read(self): return self.data
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_open(url, token, retries=3):
            return Resp(json.dumps(listing).encode() if "api.github.com" in url else b"abc")

        with mock.patch.object(github_images, "_open", fake_open), tempfile.TemporaryDirectory() as d:
            got = github_images.download_github_images("o", "r", Path(d), limit=1)
            self.assertEqual([p.name for p in got], ["a.jpg"])   # sorted, limit applied
            got = github_images.download_github_images("o", "r", Path(d))
            self.assertEqual([p.name for p in got], ["a.jpg", "b.png"])

    def test_requires_owner(self):
        with self.assertRaises(ValueError):
            github_images.list_github_images("", "")

    def test_404_message(self):
        err = urllib.error.HTTPError("u", 404, "nf", {}, None)
        with mock.patch.object(github_images, "_open", side_effect=err):
            with self.assertRaises(RuntimeError) as ctx:
                github_images.list_github_images("o", "r")
        self.assertIn("404", str(ctx.exception))


class NotebookConsistency(unittest.TestCase):
    def setUp(self):
        import nbformat
        self.nb = nbformat.read(ROOT / "WisdomToothAI.ipynb", as_version=4)

    def test_dropdown_matches_config(self):
        src = next(c.source for c in self.nb.cells if "SELECTED_MODEL =" in c.source)
        line = next(l for l in src.splitlines() if l.startswith("SELECTED_MODEL"))
        options = ast.literal_eval(line.split("#@param", 1)[1].strip())
        self.assertEqual(options, list(MODEL_OPTIONS))

    def test_code_cells_parse_and_names_exist(self):
        for cell in self.nb.cells:
            if cell.cell_type != "code":
                continue
            src = "\n".join(l for l in cell.source.splitlines() if not l.startswith("!"))
            ast.parse(src)
        import wisdomtooth.pipeline, wisdomtooth.detector, wisdomtooth.github_images  # noqa
        full = "\n".join(c.source for c in self.nb.cells if c.cell_type == "code")
        for name in ("load_model", "download_github_images", "run_pipeline", "RunSettings"):
            self.assertIn(name, full)


if __name__ == "__main__":
    unittest.main()
