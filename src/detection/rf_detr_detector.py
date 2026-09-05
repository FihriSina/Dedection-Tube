"""RF-DETR tabanlı küp dedektörü."""
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.detection.base import BaseDetector
from src.detection.models import DetectorOutput
from src.utils.types import BoundingBox, DetectionClass

# rfdetr deprecation/gelecek uyarılarını bastır (kütüphane içi, kullanıcı kodundan gelmiyor)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rfdetr")
# optimize_for_inference() sirasinda torch.jit trace uyarilari
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

logger = logging.getLogger(__name__)


class RFDETRDetector(BaseDetector):
    """RF-DETR ile küp algılama."""

    def __init__(
        self,
        model_path: str,
        model_size: str = "base",
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        resolution: int = 560,
        class_mapping: Optional[Dict[int, str]] = None,
    ):
        self.model_path = Path(model_path)
        self.model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.resolution = resolution
        self.class_mapping = class_mapping or {0: "cube"}
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            logger.error("rfdetr kütüphanesi bulunamadı. Yüklemek için: pip install rfdetr")
            return

        model_cls = RFDETRLarge if self.model_size == "large" else RFDETRBase

        if self.model_path.exists():
            try:
                self.model = model_cls(
                    pretrain_weights=str(self.model_path),
                    device=self.device,
                    resolution=self.resolution,
                    num_classes=1,
                )
                logger.info(f"RF-DETR model yüklendi: {self.model_path} (size={self.model_size})")

                # Inference hızlandırma (GPU'da FP16 Tensor Cores ile ~8x potansiyel hızlanma)
                if self.device.lower().startswith("cuda"):
                    try:
                        import torch
                        self.model.optimize_for_inference(dtype=torch.float16)
                        logger.info("RF-DETR model inference için optimize edildi (float16)")
                    except Exception as opt_e:
                        logger.warning(f"RF-DETR inference optimizasyonu atlandı: {opt_e}")
            except Exception as e:
                logger.error(f"RF-DETR model yüklenirken hata: {e}")
        else:
            logger.warning(f"RF-DETR model dosyası bulunamadı: {self.model_path}")
            logger.warning("RF-DETR dedektörü devre dışı")

    def detect(self, image: np.ndarray) -> DetectorOutput:
        """RF-DETR ile algılama yap."""
        if self.model is None:
            logger.debug("RF-DETR model yüklü değil, boş sonuç döndürülüyor")
            return DetectorOutput([], [], [], [])

        try:
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image[:, :, ::-1])  # BGR → RGB
            detections = self.model.predict(pil_image, threshold=self.confidence_threshold)
        except Exception as e:
            logger.error(f"RF-DETR inference hatası: {e}")
            return DetectorOutput([], [], [], [])

        bboxes = []
        classes = []
        confidences = []
        class_names = []

        if detections is None or len(detections) == 0:
            return DetectorOutput([], [], [], [])

        # sv.Detections arayüzü: .xyxy, .confidence, .class_id
        xyxy = detections.xyxy
        confs = detections.confidence if detections.confidence is not None else []
        cls_ids = detections.class_id if detections.class_id is not None else []

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i]) if i < len(confs) else self.confidence_threshold
            cls_id = int(cls_ids[i]) if i < len(cls_ids) else 0
            class_name = self.class_mapping.get(cls_id, "cube")

            bboxes.append(BoundingBox(int(x1), int(y1), int(x2), int(y2)))
            classes.append(self._map_class_name(class_name))
            confidences.append(conf)
            class_names.append(class_name)

        logger.debug(f"RF-DETR dedektör: {len(bboxes)} nesne bulundu")
        return DetectorOutput(bboxes, classes, confidences, class_names)

    def _map_class_name(self, class_name: str) -> DetectionClass:
        if "cube" in class_name.lower():
            return DetectionClass.CUBE
        return DetectionClass.CUBE

    def get_name(self) -> str:
        return "RFDETRDetector"

    @classmethod
    def from_config(cls, config: "DetectionConfig") -> "RFDETRDetector":
        """Konfigürasyondan dedektör oluştur."""
        rf = config.rf_detr
        return cls(
            model_path=rf.model_path,
            model_size=rf.model_size,
            confidence_threshold=rf.confidence_threshold,
            device=rf.device,
            resolution=rf.resolution,
            class_mapping={int(k): v for k, v in rf.classes.items()},
        )
