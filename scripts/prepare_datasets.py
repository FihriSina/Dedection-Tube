"""Yerel cube datasetlerini birleştirir ve RF-DETR için COCO dataseti üretir."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "data.yaml"

UNIFIED_CLASS_NAMES: List[str] = []
UNIFIED_CLASS_IDS: Dict[str, int] = {}

# RF-DETR "valid" beklediğinden "val" adını eşleştiriyoruz.
COCO_SPLIT_MAP: Dict[str, str] = {"train": "train", "val": "valid"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_FALLBACK_DATASET_DIRS = (ROOT / "data" / "yolo",)
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
}


@dataclass
class DatasetSource:
    name: str
    source_type: str
    path: Path


@dataclass
class DatasetEntry:
    source_name: str
    source_root: Path
    image_path: Path
    label_path: Path
    source_split: str
    width: int
    height: int
    label_counts: Counter


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _init_class_names(cfg: dict) -> None:
    """Config'den sınıf adlarını ve ID eşlemesini başlat."""
    global UNIFIED_CLASS_NAMES, UNIFIED_CLASS_IDS
    classes_cfg = cfg.get("classes", {})
    hierarchy = classes_cfg.get("hierarchy", {})
    subclasses: List[str] = hierarchy.get("subclasses", [])
    if subclasses:
        UNIFIED_CLASS_NAMES = subclasses
    else:
        parent = hierarchy.get("parent", "cube")
        UNIFIED_CLASS_NAMES = [parent]
    UNIFIED_CLASS_IDS = {name: idx for idx, name in enumerate(UNIFIED_CLASS_NAMES)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def detect_dataset_root(base_dir: Path) -> Optional[Path]:
    candidates = [base_dir] + [path for path in base_dir.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "data.yaml").exists():
            return candidate
    return None


def read_dataset_yaml(dataset_root: Path) -> dict:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = dataset_root / "dataset.yaml"
    if not data_yaml.exists():
        images_root = dataset_root / "images"
        labels_root = dataset_root / "labels"
        if images_root.exists() and labels_root.exists():
            return {
                "path": str(dataset_root.as_posix()),
                "train": "images",
                "names": ["cube"],
            }
        raise FileNotFoundError(f"data.yaml bulunamadı: {dataset_root}")
    with data_yaml.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_split_name(name: str) -> Optional[str]:
    return SPLIT_ALIASES.get(name.strip().lower())


def resolve_split_dirs(dataset_root: Path, dataset_yaml: dict) -> Dict[str, Tuple[Path, Path]]:
    split_dirs: Dict[str, Tuple[Path, Path]] = {}

    for key in ("train", "val", "test"):
        if key not in dataset_yaml:
            continue
        raw_value = str(dataset_yaml[key]).replace("\\", "/")
        image_dir = (dataset_root / raw_value).resolve() if not Path(raw_value).is_absolute() else Path(raw_value)
        label_dir = Path(str(image_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        if (not label_dir.exists() or label_dir == image_dir) and image_dir.name == "images":
            label_dir = image_dir.parent / "labels"
        if not label_dir.exists() and image_dir.parent.name == "images":
            label_dir = image_dir.parent.parent / "labels" / image_dir.name
        if image_dir.exists() and label_dir.exists():
            split_dirs[key] = (image_dir, label_dir)

    if split_dirs:
        return split_dirs

    # Try nested structure: split_name/images and split_name/labels
    for split_name in ("train", "valid", "validation", "val", "test", "testing"):
        split = normalize_split_name(split_name)
        if not split:
            continue
        image_dir = dataset_root / split_name / "images"
        label_dir = dataset_root / split_name / "labels"
        if image_dir.exists() and label_dir.exists():
            split_dirs[split] = (image_dir, label_dir)

    if split_dirs:
        return split_dirs

    # Try flat structure: images/ and labels/ at root
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if images_root.exists() and labels_root.exists():
        for child in images_root.iterdir():
            if not child.is_dir():
                continue
            split = normalize_split_name(child.name)
            if not split:
                continue
            label_dir = labels_root / child.name
            if label_dir.exists():
                split_dirs[split] = (child, label_dir)

        if split_dirs:
            return split_dirs

        has_images = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in images_root.iterdir()
        )
        has_labels = any(path.is_file() and path.suffix.lower() == ".txt" for path in labels_root.iterdir())
        if has_images and has_labels:
            split_dirs["train"] = (images_root, labels_root)

    return split_dirs


OBSTACLE_TOKENS = ("hole", "cylinder", "sticker", "arm", "roboticarm", "obstacle")


def normalize_label_name(raw_name: str) -> Optional[str]:
    """Kaynak etiketlerini tek 'cube' sınıfına indirger; engel etiketlerini atar."""
    lowered = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in lowered for token in OBSTACLE_TOKENS):
        return None
    return "cube"


def dataset_class_map(dataset_yaml: dict) -> Dict[int, Optional[int]]:
    raw_names = dataset_yaml.get("names", [])
    if isinstance(raw_names, dict):
        iterator = raw_names.items()
    else:
        iterator = enumerate(raw_names)

    mapping: Dict[int, Optional[int]] = {}
    for raw_index, raw_name in iterator:
        unified = normalize_label_name(str(raw_name))
        mapping[int(raw_index)] = UNIFIED_CLASS_IDS[unified] if unified is not None else None
    return mapping


def collect_image_files(image_dir: Path) -> Dict[str, Path]:
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def validate_image(image_path: Path) -> Optional[Tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return None
    return width, height


def remap_label_file(source_label: Path, destination_label: Optional[Path], class_map: Dict[int, Optional[int]]) -> Tuple[Counter, bool]:
    counts: Counter = Counter()
    rewritten_lines: List[str] = []

    with source_label.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                return Counter(), False
            try:
                source_class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:5]]
            except ValueError:
                return Counter(), False
            if source_class_id not in class_map:
                return Counter(), False
            if class_map[source_class_id] is None:
                continue
            x_center, y_center, width, height = coords
            if not (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                return Counter(), False
            target_class_id = class_map[source_class_id]
            assert target_class_id is not None
            parts[0] = str(target_class_id)
            rewritten_lines.append(" ".join(parts))
            counts[UNIFIED_CLASS_NAMES[target_class_id]] += 1

    if not rewritten_lines:
        return Counter(), False
    if destination_label is not None:
        destination_label.write_text("\n".join(rewritten_lines), encoding="utf-8")
    return counts, True


def inspect_and_collect_dataset(source: DatasetSource, stats: dict) -> List[DatasetEntry]:
    dataset_yaml = read_dataset_yaml(source.path)
    raw_names = dataset_yaml.get("names", [])
    raw_class_names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)

    split_dirs = resolve_split_dirs(source.path, dataset_yaml)
    if not split_dirs:
        raise RuntimeError(f"Dataset split dizinleri bulunamadı: {source.path}")

    class_map = dataset_class_map(dataset_yaml)
    source_stats = stats["sources"].setdefault(
        source.name,
        {
            "type": source.source_type,
            "included": True,
            "raw_classes": raw_class_names,
            "images": 0,
            "labels": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "split_images": defaultdict(int),
            "source_split_images": defaultdict(int),
            "class_distribution": Counter(),
            "image_dimensions": Counter(),
            "label_format": "yolo",
        },
    )

    entries: List[DatasetEntry] = []
    for split, (image_dir, label_dir) in split_dirs.items():
        image_files = collect_image_files(image_dir)

        for label_path in label_dir.glob("*.txt"):
            source_stats["labels"] += 1
            image_path = image_files.get(label_path.stem)
            if image_path is None:
                source_stats["invalid_images"] += 1
                stats["totals"]["skipped_missing_image"] += 1
                continue

            source_stats["images"] += 1
            image_size = validate_image(image_path)
            if image_size is None:
                source_stats["invalid_images"] += 1
                stats["totals"]["skipped_invalid_image"] += 1
                continue

            label_counts, is_valid = remap_label_file(label_path, None, class_map)
            if not is_valid:
                source_stats["invalid_images"] += 1
                stats["totals"]["skipped_invalid_label"] += 1
                continue

            width, height = image_size
            source_stats["valid_images"] += 1
            source_stats["source_split_images"][split] += 1
            source_stats["class_distribution"].update(label_counts)
            source_stats["image_dimensions"][f"{width}x{height}"] += 1

            entries.append(
                DatasetEntry(
                    source_name=source.name,
                    source_root=source.path,
                    image_path=image_path,
                    label_path=label_path,
                    source_split=split,
                    width=width,
                    height=height,
                    label_counts=label_counts,
                )
            )

    return entries


def split_entries(entries: List[DatasetEntry], cfg: dict) -> Dict[str, List[DatasetEntry]]:
    data_cfg = cfg.get("data", {})
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))

    base_cfg = {}
    try:
        import sys
        base_config_path = ROOT / "config" / "base.yaml"
        with base_config_path.open("r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    random_seed = int((base_cfg.get("system") or {}).get("random_seed", 42))

    shuffled = entries[:]
    random.Random(random_seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_unified_dataset(entries_by_split: Dict[str, List[DatasetEntry]], output_root: Path, stats: dict) -> None:
    class_maps: Dict[Path, Dict[int, int]] = {}
    for split, entries in entries_by_split.items():
        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split
        ensure_dir(output_image_dir)
        ensure_dir(output_label_dir)

        for entry in entries:
            target_name = f"{entry.source_name}_{entry.source_split}_{entry.image_path.stem}"
            destination_image = output_image_dir / f"{target_name}{entry.image_path.suffix.lower()}"
            destination_label = output_label_dir / f"{target_name}.txt"

            if entry.source_root not in class_maps:
                class_maps[entry.source_root] = dataset_class_map(read_dataset_yaml(entry.source_root))

            shutil.copy2(entry.image_path, destination_image)
            remap_label_file(entry.label_path, destination_label, class_maps[entry.source_root])

            source_stats = stats["sources"][entry.source_name]
            source_stats["split_images"][split] += 1

            stats["totals"]["images"] += 1
            stats["totals"]["labels"] += 1
            stats["totals"]["split_images"][split] += 1
            stats["totals"]["class_distribution"].update(entry.label_counts)
            stats["totals"]["image_dimensions"][f"{entry.width}x{entry.height}"] += 1


def build_data_yaml(output_root: Path) -> dict:
    data_yaml = {
        "path": str(output_root.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(UNIFIED_CLASS_NAMES),
        "names": UNIFIED_CLASS_NAMES,
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, allow_unicode=True, sort_keys=False)
    return data_yaml


def convert_to_coco(unified_root: Path, coco_root: Path) -> Dict[str, int]:
    """Unified dataseti RF-DETR için COCO formatına çevirir.

    unified_root: data/processed/unified (images/ ve labels/ içerir)
    coco_root:    data/processed/coco    (train/ ve valid/ oluşturulur)
    """
    categories = [
        {"id": i + 1, "name": name, "supercategory": name}
        for i, name in enumerate(UNIFIED_CLASS_NAMES)
    ]
    written: Dict[str, int] = {}
    for yolo_split, coco_split in COCO_SPLIT_MAP.items():
        image_dir = unified_root / "images" / yolo_split
        label_dir = unified_root / "labels" / yolo_split
        if not image_dir.exists() or not label_dir.exists():
            continue
        split_out = coco_root / coco_split
        ensure_dir(split_out)
        image_files = collect_image_files(image_dir)
        images_list: list = []
        annotations_list: list = []
        image_id = 1
        ann_id = 1
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_files.get(label_path.stem)
            if image_path is None:
                continue
            image_size = validate_image(image_path)
            if image_size is None:
                continue
            img_w, img_h = image_size
            shutil.copy2(image_path, split_out / image_path.name)
            images_list.append({"id": image_id, "file_name": image_path.name,
                                 "width": img_w, "height": img_h})
            with label_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = (float(v) for v in parts[1:5])
                    x_min = (xc - bw / 2) * img_w
                    y_min = (yc - bh / 2) * img_h
                    w_px, h_px = bw * img_w, bh * img_h
                    annotations_list.append({
                        "id": ann_id, "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)],
                        "area": round(w_px * h_px, 2), "iscrowd": 0,
                    })
                    ann_id += 1
            image_id += 1
        ann_path = split_out / "_annotations.coco.json"
        ann_path.write_text(
            json.dumps({"images": images_list, "annotations": annotations_list,
                        "categories": categories}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written[coco_split] = len(images_list)
        print(f"[INFO] COCO {coco_split}: {len(images_list)} görüntü, {len(annotations_list)} annotation → {split_out}")
    return written


def finalize_stats(stats: dict) -> dict:
    total_images = stats["totals"]["images"]
    split_images = stats["totals"]["split_images"]
    split_ratios = {
        split: round((count / total_images), 4) if total_images else 0.0
        for split, count in split_images.items()
    }

    serializable_sources = {}
    for name, source_stats in stats["sources"].items():
        serializable_sources[name] = {
            "type": source_stats["type"],
            "included": source_stats["included"],
            "raw_classes": source_stats["raw_classes"],
            "images": source_stats["images"],
            "labels": source_stats["labels"],
            "valid_images": source_stats["valid_images"],
            "invalid_images": source_stats["invalid_images"],
            "split_images": dict(source_stats["split_images"]),
            "source_split_images": dict(source_stats["source_split_images"]),
            "class_distribution": dict(source_stats["class_distribution"]),
            "image_dimensions": dict(source_stats["image_dimensions"]),
            "label_format": source_stats["label_format"],
        }

    return {
        "totals": {
            "images": total_images,
            "labels": stats["totals"]["labels"],
            "split_images": dict(split_images),
            "split_ratios": split_ratios,
            "class_distribution": dict(stats["totals"]["class_distribution"]),
            "image_dimensions": dict(stats["totals"]["image_dimensions"]),
            "skipped_missing_image": stats["totals"]["skipped_missing_image"],
            "skipped_invalid_image": stats["totals"]["skipped_invalid_image"],
            "skipped_invalid_label": stats["totals"]["skipped_invalid_label"],
        },
        "sources": serializable_sources,
        "classes": UNIFIED_CLASS_NAMES,
    }


def write_stats(output_root: Path, stats: dict) -> None:
    finalized = finalize_stats(stats)
    stats_path = output_root / "stats.json"
    stats_path.write_text(json.dumps(finalized, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Unified Dataset Report",
        "",
        f"- Toplam görüntü: {finalized['totals']['images']}",
        f"- Toplam etiket dosyası: {finalized['totals']['labels']}",
        f"- Split dağılımı: {finalized['totals']['split_images']}",
        f"- Split oranları: {finalized['totals']['split_ratios']}",
        f"- Sınıf dağılımı: {finalized['totals']['class_distribution']}",
        f"- Görüntü boyutları: {finalized['totals']['image_dimensions']}",
        f"- Atlanan eksik görüntü: {finalized['totals']['skipped_missing_image']}",
        f"- Atlanan bozuk görüntü: {finalized['totals']['skipped_invalid_image']}",
        f"- Atlanan bozuk etiket: {finalized['totals']['skipped_invalid_label']}",
        "",
        "## Kaynak bazlı özet",
        "",
    ]
    for source_name, source_stats in finalized["sources"].items():
        lines.append(f"### {source_name}")
        lines.append(f"- Tip: {source_stats['type']}")
        lines.append(f"- Ham sınıflar: {source_stats['raw_classes']}")
        lines.append(f"- Görüntü: {source_stats['images']}")
        lines.append(f"- Geçerli görüntü: {source_stats['valid_images']}")
        lines.append(f"- Geçersiz görüntü/etiket: {source_stats['invalid_images']}")
        lines.append(f"- Kaynak split: {source_stats['source_split_images']}")
        lines.append(f"- Unified split: {source_stats['split_images']}")
        lines.append(f"- Sınıf dağılımı: {source_stats['class_distribution']}")
        lines.append(f"- Görüntü boyutları: {source_stats['image_dimensions']}")
        lines.append(f"- Etiket formatı: {source_stats['label_format']}")
        lines.append("")

    (output_root / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def initialize_stats() -> dict:
    return {
        "totals": {
            "images": 0,
            "labels": 0,
            "split_images": Counter(),
            "class_distribution": Counter(),
            "image_dimensions": Counter(),
            "skipped_missing_image": 0,
            "skipped_invalid_image": 0,
            "skipped_invalid_label": 0,
        },
        "sources": {},
    }


def discover_sources(raw_dir: Path) -> List[DatasetSource]:
    sources: List[DatasetSource] = []
    seen_roots: set = set()
    for yaml_name in ("data.yaml", "dataset.yaml"):
        for data_yaml in raw_dir.rglob(yaml_name):
            source_root = data_yaml.parent
            if source_root in seen_roots:
                continue
            seen_roots.add(source_root)
            source_name = source_root.name.replace(" ", "_").lower()
            source_type = "synthetic" if "synthetic" in source_root.as_posix().lower() else "local"
            sources.append(DatasetSource(name=source_name, source_type=source_type, path=source_root))

    for fallback_dir in DEFAULT_FALLBACK_DATASET_DIRS:
        if not fallback_dir.exists():
            continue
        images_root = fallback_dir / "images"
        labels_root = fallback_dir / "labels"
        if not (images_root.exists() and labels_root.exists()):
            continue
        source_name = fallback_dir.name.replace(" ", "_").lower()
        sources.append(DatasetSource(name=source_name, source_type="local", path=fallback_dir))

    unique: Dict[str, DatasetSource] = {}
    for source in sources:
        unique[str(source.path)] = source
    return list(unique.values())


def prepare_output_dirs(unified_root: Path) -> None:
    remove_dir(unified_root)
    for split in ("train", "val", "test"):
        ensure_dir(unified_root / "images" / split)
        ensure_dir(unified_root / "labels" / split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cube dataset birleştirici")
    parser.add_argument(
        "--coco",
        action="store_true",
        default=False,
        help="Unified dataseti COCO formatına çevir (data/processed/coco/ oluşturur)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    _init_class_names(config)
    data_cfg = config["data"]
    raw_dir = ROOT / data_cfg["raw_dir"]
    unified_root = ROOT / data_cfg["unified_dir"]

    ensure_dir(raw_dir)
    prepare_output_dirs(unified_root)

    sources = discover_sources(raw_dir)

    if not sources:
        print(f"[ERROR] {raw_dir} altında hiçbir dataset bulunamadı.")
        return 1

    stats = initialize_stats()
    collected_entries: List[DatasetEntry] = []
    for source in sources:
        print(f"[INFO] İnceleniyor: {source.name} -> {source.path}")
        collected_entries.extend(inspect_and_collect_dataset(source, stats))

    if not collected_entries:
        print("[ERROR] Geçerli görüntü/etiket çifti bulunamadı.")
        return 1

    split_map = split_entries(collected_entries, config)
    write_unified_dataset(split_map, unified_root, stats)
    build_data_yaml(unified_root)
    write_stats(unified_root, stats)

    if args.coco:
        coco_root = ROOT / data_cfg["processed_dir"] / "coco"
        convert_to_coco(unified_root, coco_root)

    finalized = finalize_stats(stats)
    print("[INFO] Unified dataset hazırlandı.")
    print(json.dumps(finalized["totals"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())