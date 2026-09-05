"""Dedektör fusion - RF-DETR sinir ağı ile klasik CV birleştir"""
import cv2
import numpy as np
import logging

from src.detection.base import BaseDetector
from src.detection.models import DetectorOutput
from src.detection.rf_detr_detector import RFDETRDetector
from src.detection.classical_detector import ClassicalDetector
from src.utils.types import BoundingBox

logger = logging.getLogger(__name__)


class FusionDetector(BaseDetector):
    """Sinir ağı dedektörünü klasik CV ile birleştiren hibrit dedektör."""

    def __init__(
        self,
        neural_detector: BaseDetector,
        classical_detector: ClassicalDetector,
        prefer_neural: bool = True,
        fallback_if_neural_fails: bool = True,
        fallback_if_low_confidence: bool = True,
        fallback_confidence_threshold: float = 0.4,
        use_nms: bool = True,
        nms_iou_threshold: float = 0.5,
    ):
        self.neural = neural_detector
        self.classical = classical_detector
        self.prefer_neural = prefer_neural
        self.fallback_if_neural_fails = fallback_if_neural_fails
        self.fallback_if_low_confidence = fallback_if_low_confidence
        self.fallback_confidence_threshold = fallback_confidence_threshold
        self.use_nms = use_nms
        self.nms_iou_threshold = nms_iou_threshold

    def detect(self, image: np.ndarray) -> DetectorOutput:
        neural_output = self.neural.detect(image)

        use_classical = False

        if self.fallback_if_neural_fails and neural_output.is_empty():
            logger.debug("Sinir ağı boş sonuç, klasik dedektör devreye giriyor")
            use_classical = True
        elif self.fallback_if_low_confidence and not neural_output.is_empty():
            low_conf_count = sum(
                1 for conf in neural_output.confidences
                if conf < self.fallback_confidence_threshold
            )
            if low_conf_count > 0:
                logger.debug(f"{low_conf_count} düşük güvenli algılama, klasik ekleniyor")
                use_classical = True

        classical_output = self.classical.detect(image) if use_classical else DetectorOutput([], [], [], [])

        if self.prefer_neural:
            merged = self._merge_detections(neural_output, classical_output)
        else:
            merged = self._merge_detections(classical_output, neural_output)

        if self.use_nms and not merged.is_empty():
            merged = self._apply_nms(merged)

        logger.debug(
            f"Fusion sonuç: {len(merged)} algılama "
            f"({self.neural.get_name()}: {len(neural_output)}, Klasik: {len(classical_output)})"
        )

        return merged

    def _merge_detections(
        self,
        primary: DetectorOutput,
        secondary: DetectorOutput,
    ) -> DetectorOutput:
        bboxes = primary.bboxes.copy()
        classes = primary.classes.copy()
        confidences = primary.confidences.copy()
        class_names = primary.class_names.copy()

        for i in range(len(secondary)):
            bbox = secondary.bboxes[i]
            has_overlap = any(
                self._compute_iou(bbox, pb) > self.nms_iou_threshold
                for pb in primary.bboxes
            )
            if not has_overlap:
                bboxes.append(bbox)
                classes.append(secondary.classes[i])
                confidences.append(secondary.confidences[i])
                class_names.append(secondary.class_names[i])

        return DetectorOutput(bboxes, classes, confidences, class_names)

    def _apply_nms(self, detections: DetectorOutput) -> DetectorOutput:
        if len(detections) == 0:
            return detections

        boxes = np.array(
            [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in detections.bboxes],
            dtype=np.float32,
        )
        scores = np.array(detections.confidences, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=self.nms_iou_threshold,
        )

        if len(indices) == 0:
            return DetectorOutput([], [], [], [])

        indices = indices.flatten()
        return DetectorOutput(
            bboxes=[detections.bboxes[i] for i in indices],
            classes=[detections.classes[i] for i in indices],
            confidences=[detections.confidences[i] for i in indices],
            class_names=[detections.class_names[i] for i in indices],
        )

    @staticmethod
    def _compute_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection
        return intersection / union if union > 0 else 0.0

    def get_name(self) -> str:
        return f"FusionDetector({self.neural.get_name()}+Classical)"

    @classmethod
    def from_config(cls, config: "DetectionConfig") -> "FusionDetector":
        """Konfigürasyondan RF-DETR + klasik CV fusion dedektör oluştur."""
        neural = RFDETRDetector.from_config(config)
        try:
            logger.info(f"Seçilen neural dedektör: {neural.get_name()}")
        except Exception:
            logger.debug("Neural dedektör adı alınamadı")

        classical = ClassicalDetector.from_config(config)
        f = config.fusion
        return cls(
            neural_detector=neural,
            classical_detector=classical,
            prefer_neural=f.prefer_neural,
            fallback_if_neural_fails=f.fallback_if_neural_fails,
            fallback_if_low_confidence=f.fallback_if_low_confidence,
            fallback_confidence_threshold=f.fallback_confidence_threshold,
            use_nms=f.use_nms,
            nms_iou_threshold=f.nms_iou_threshold,
        )
