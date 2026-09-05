"""BoxMOT takip adaptörü.

Bu modül, projenin DetectorOutput formatını BoxMOT'un frame bazlı
``[x1, y1, x2, y2, conf, cls]`` girişine çevirir ve dönen track ID'lerini
tekrar DetectorOutput.track_ids alanına hizalar.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol

import numpy as np

from src.detection.models import DetectorOutput
from src.utils.types import BoundingBox, DetectionClass

logger = logging.getLogger(__name__)


class TrackingAdapter(Protocol):
    """Pipeline içindeki takip adaptörlerinin ortak arayüzü."""

    def update(self, image: np.ndarray, detections: DetectorOutput) -> DetectorOutput:
        """Bir frame için tespitleri takip ID'leriyle güncelle."""
        ...


class NullTracker:
    """Tracking kapalıyken kullanılan no-op adaptör."""

    enabled = False

    def update(self, image: np.ndarray, detections: DetectorOutput) -> DetectorOutput:
        if len(detections.track_ids) == len(detections):
            return detections
        return detections.with_track_ids([None] * len(detections))


class BoxMOTTracker:
    """BoxMOT tracker'ını mevcut küp tespit pipeline'ına bağlar."""

    enabled = True

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        tracker_config: Optional[str] = None,
        reid_weights: Optional[str] = None,
        device: Optional[str] = None,
        half: bool = False,
        per_class: bool = True,
        tracker_backend: str = "python",
        keep_untracked_detections: bool = True,
    ) -> None:
        self.tracker_type = tracker_type.lower().strip()
        self.keep_untracked_detections = keep_untracked_detections

        try:
            from boxmot.trackers.tracker_zoo import create_tracker
        except ImportError as exc:
            raise RuntimeError(
                "BoxMOT yüklü değil. Kurulum için: pip install boxmot "
                "veya proje bağımlılıklarını yeniden yükleyin."
            ) from exc

        self._tracker = create_tracker(
            tracker_type=self.tracker_type,
            tracker_config=tracker_config,
            reid_weights=reid_weights,
            device=device,
            half=half,
            per_class=per_class,
            tracker_backend=tracker_backend,
        )
        logger.info("BoxMOT tracker hazır: %s (%s backend)", self.tracker_type, tracker_backend)

    @classmethod
    def from_config(cls, config: Any) -> TrackingAdapter:
        """Pydantic tracking config nesnesinden adaptör oluştur."""
        if not getattr(config, "enabled", False):
            return NullTracker()

        return cls(
            tracker_type=getattr(config, "tracker_type", "bytetrack"),
            tracker_config=getattr(config, "tracker_config", None),
            reid_weights=getattr(config, "reid_weights", None),
            device=getattr(config, "device", None),
            half=bool(getattr(config, "half", False)),
            per_class=bool(getattr(config, "per_class", True)),
            tracker_backend=getattr(config, "tracker_backend", "python"),
            keep_untracked_detections=bool(getattr(config, "keep_untracked_detections", True)),
        )

    def update(self, image: np.ndarray, detections: DetectorOutput) -> DetectorOutput:
        """BoxMOT'u bir frame ilerlet ve tespitlere kalıcı track ID ata."""
        det_array = self._detections_to_boxmot_array(detections)
        tracks_raw = self._tracker.update(det_array, image)
        tracks = self._tracks_to_array(tracks_raw)

        if detections.is_empty():
            return detections.with_track_ids([])

        tracked_by_det = self._build_tracked_detection_map(
            tracks=tracks,
            detections=detections,
            image_shape=image.shape,
        )

        bboxes = []
        classes = []
        confidences = []
        class_names = []
        track_ids = []

        for det_index in range(len(detections)):
            tracked = tracked_by_det.get(det_index)
            if tracked is not None:
                bboxes.append(tracked["bbox"])
                classes.append(detections.classes[det_index])
                confidences.append(tracked["confidence"])
                class_names.append(detections.class_names[det_index])
                track_ids.append(tracked["track_id"])
            elif self.keep_untracked_detections:
                bboxes.append(detections.bboxes[det_index])
                classes.append(detections.classes[det_index])
                confidences.append(detections.confidences[det_index])
                class_names.append(detections.class_names[det_index])
                track_ids.append(None)

        return DetectorOutput(
            bboxes=bboxes,
            classes=classes,
            confidences=confidences,
            class_names=class_names,
            track_ids=track_ids,
        )

    @staticmethod
    def _detections_to_boxmot_array(detections: DetectorOutput) -> np.ndarray:
        rows = []
        for bbox, cls, conf in zip(detections.bboxes, detections.classes, detections.confidences):
            rows.append(
                [
                    float(bbox.x1),
                    float(bbox.y1),
                    float(bbox.x2),
                    float(bbox.y2),
                    float(conf),
                    float(_class_to_boxmot_id(cls)),
                ]
            )

        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _tracks_to_array(tracks: Any) -> np.ndarray:
        if tracks is None:
            return np.empty((0, 8), dtype=np.float32)

        arr = np.asarray(tracks)
        if arr.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32, copy=False)

    def _build_tracked_detection_map(
        self,
        tracks: np.ndarray,
        detections: DetectorOutput,
        image_shape: tuple[int, ...],
    ) -> Dict[int, Dict[str, Any]]:
        tracked_by_det: Dict[int, Dict[str, Any]] = {}
        used_det_indices: set[int] = set()

        for track_row in tracks:
            if track_row.shape[0] < 7:
                continue

            track_id = int(round(float(track_row[4])))
            if track_id < 0:
                continue

            det_index = self._extract_det_index(track_row)
            if det_index is None:
                det_index = self._match_track_to_detection(track_row, detections, used_det_indices)

            if det_index is None or det_index < 0 or det_index >= len(detections):
                # BoxMOT coasting track'leri det_ind=-1 ile döndürebilir; bunlar gerçek
                # tespit olmadığı için pose pipeline'ına sokulmuyor.
                continue
            if det_index in tracked_by_det:
                continue

            bbox = self._bbox_from_track_row(track_row, image_shape)
            if bbox is None:
                continue

            confidence = float(track_row[5]) if np.isfinite(track_row[5]) else detections.confidences[det_index]
            tracked_by_det[det_index] = {
                "bbox": bbox,
                "confidence": confidence,
                "track_id": track_id,
            }
            used_det_indices.add(det_index)

        return tracked_by_det

    @staticmethod
    def _extract_det_index(track_row: np.ndarray) -> Optional[int]:
        # BoxMOT 21.x TrackResults formatı: [x1, y1, x2, y2, id, conf, cls, det_ind]
        if track_row.shape[0] >= 8:
            return int(round(float(track_row[7])))
        return None

    @staticmethod
    def _bbox_from_track_row(track_row: np.ndarray, image_shape: tuple[int, ...]) -> Optional[BoundingBox]:
        height, width = image_shape[:2]
        x1 = int(round(float(track_row[0])))
        y1 = int(round(float(track_row[1])))
        x2 = int(round(float(track_row[2])))
        y2 = int(round(float(track_row[3])))

        x1 = max(0, min(x1, max(width - 1, 0)))
        y1 = max(0, min(y1, max(height - 1, 0)))
        x2 = max(0, min(x2, max(width - 1, 0)))
        y2 = max(0, min(y2, max(height - 1, 0)))

        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _match_track_to_detection(
        track_row: np.ndarray,
        detections: DetectorOutput,
        used_det_indices: set[int],
    ) -> Optional[int]:
        # Geriye dönük uyumluluk: det_ind olmayan BoxMOT sürümünde IoU ile eşleştir.
        track_bbox = BoundingBox(
            x1=int(round(float(track_row[0]))),
            y1=int(round(float(track_row[1]))),
            x2=int(round(float(track_row[2]))),
            y2=int(round(float(track_row[3]))),
        )
        best_index = None
        best_iou = 0.0
        for index, det_bbox in enumerate(detections.bboxes):
            if index in used_det_indices:
                continue
            iou = _iou(track_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        return best_index if best_iou >= 0.3 else None


def build_tracker_from_config(config: Any) -> TrackingAdapter:
    """Tracking config üzerinden uygun adaptörü oluştur."""
    return BoxMOTTracker.from_config(config)


def _class_to_boxmot_id(cls: Any) -> int:
    if cls == DetectionClass.CUBE:
        return 0
    try:
        return int(cls)
    except (TypeError, ValueError):
        return 0


def _iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    union = bbox1.area + bbox2.area - intersection
    return intersection / union if union > 0 else 0.0
