from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_datasets.py"
SPEC = importlib.util.spec_from_file_location("prepare_datasets", MODULE_PATH)
prepare_datasets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare_datasets
SPEC.loader.exec_module(prepare_datasets)


def test_read_dataset_yaml_supports_flat_yolo_layout_without_yaml(tmp_path):
    dataset_root = tmp_path / "data" / "yolo"
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    images_root.mkdir(parents=True)
    labels_root.mkdir(parents=True)

    dataset_yaml = prepare_datasets.read_dataset_yaml(dataset_root)

    assert dataset_yaml["train"] == "images"
    assert dataset_yaml["names"] == ["cube"]


def test_resolve_split_dirs_supports_flat_yolo_layout_without_split_subdirs(tmp_path):
    dataset_root = tmp_path / "data" / "yolo"
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    images_root.mkdir(parents=True)
    labels_root.mkdir(parents=True)
    (images_root / "sample.jpg").write_bytes(b"fake")
    (labels_root / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    split_dirs = prepare_datasets.resolve_split_dirs(dataset_root, {"train": "images", "names": ["cube"]})

    assert split_dirs == {"train": (images_root, labels_root)}


def test_discover_sources_includes_default_flat_yolo_dir(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    yolo_root = tmp_path / "data" / "yolo"
    (yolo_root / "images").mkdir(parents=True)
    (yolo_root / "labels").mkdir(parents=True)

    monkeypatch.setattr(prepare_datasets, "DEFAULT_FALLBACK_DATASET_DIRS", (yolo_root,))

    sources = prepare_datasets.discover_sources(raw_dir)

    assert len(sources) == 1
    assert sources[0].name == "yolo"
    assert sources[0].path == yolo_root


def _make_split(tmp_path: Path, split: str, label_line: str = "0 0.5 0.5 0.4 0.4\n",
                img_size: tuple = (10, 10)) -> None:
    img_bytes = cv2.imencode(".jpg", np.zeros((*img_size[::-1], 3), dtype=np.uint8))[1].tobytes()
    (tmp_path / "images" / split).mkdir(parents=True)
    (tmp_path / "labels" / split).mkdir(parents=True)
    (tmp_path / "images" / split / "img1.jpg").write_bytes(img_bytes)
    (tmp_path / "labels" / split / "img1.txt").write_text(label_line, encoding="utf-8")


def test_convert_to_coco_creates_valid_not_val(tmp_path, monkeypatch):
    for split in ("train", "val"):
        _make_split(tmp_path, split)
    monkeypatch.setattr(prepare_datasets, "UNIFIED_CLASS_NAMES", ["cube"])
    prepare_datasets.convert_to_coco(tmp_path, tmp_path / "coco")
    assert (tmp_path / "coco" / "valid").is_dir()
    assert not (tmp_path / "coco" / "val").exists()
    assert (tmp_path / "coco" / "train" / "_annotations.coco.json").exists()
    assert (tmp_path / "coco" / "valid" / "_annotations.coco.json").exists()


def test_convert_to_coco_bbox_pixel_format(tmp_path, monkeypatch):
    # img: width=200, height=100; YOLO: xc=0.5 yc=0.5 bw=0.4 bh=0.2
    _make_split(tmp_path, "train", label_line="0 0.5 0.5 0.4 0.2\n", img_size=(200, 100))
    monkeypatch.setattr(prepare_datasets, "UNIFIED_CLASS_NAMES", ["cube"])
    prepare_datasets.convert_to_coco(tmp_path, tmp_path / "coco")
    ann = json.loads((tmp_path / "coco" / "train" / "_annotations.coco.json").read_text())
    # x_min=(0.5-0.2)*200=60, y_min=(0.5-0.1)*100=40, w=0.4*200=80, h=0.2*100=20
    assert ann["annotations"][0]["bbox"] == [60.0, 40.0, 80.0, 20.0]


def test_convert_to_coco_category_id_one_indexed(tmp_path, monkeypatch):
    _make_split(tmp_path, "train", label_line="0 0.5 0.5 0.3 0.3\n")
    monkeypatch.setattr(prepare_datasets, "UNIFIED_CLASS_NAMES", ["cube"])
    prepare_datasets.convert_to_coco(tmp_path, tmp_path / "coco")
    ann = json.loads((tmp_path / "coco" / "train" / "_annotations.coco.json").read_text())
    assert ann["annotations"][0]["category_id"] == 1
    assert ann["categories"][0]["id"] == 1
    assert ann["categories"][0]["name"] == "cube"