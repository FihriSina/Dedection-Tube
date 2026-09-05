"""Klasik bilgisayarlı görü tabanlı küp dedektörü (HSV + kontur)"""
# classical_detector.py, HSV renk maskesi ve kontur analizi kullanarak küp tespiti yapan 
# bir dedektör sınıfını içerir. Bu sınıf, BaseDetector'ı genişleterek detect ve get_name yöntemlerini uygular. 
# detect yöntemi, verilen bir görüntü üzerinde ön işleme, kenar tespiti ve kontur analizi yaparak küp adaylarını 
# değerlendirir ve sonuçları DetectorOutput formatında döndürür. get_name yöntemi ise dedektörün adını sağlar. 
# Bu dedektör, pipeline içinde daha basit ve hızlı bir alternatif olarak kullanılabilir.
import cv2
import numpy as np
import logging
from typing import Tuple

from src.detection.base import BaseDetector
from src.detection.models import DetectorOutput
from src.utils.types import BoundingBox, DetectionClass

logger = logging.getLogger(__name__)


class ClassicalDetector(BaseDetector):
    """HSV renk maskesi + kontur analizi ile küp algılama"""
    
    def __init__(
        self,
        blur_kernel: int = 5,
        morph_kernel: int = 5,
        min_area: int = 500,
        max_area: int = 100000,
        min_aspect_ratio: float = 0.7,
        max_aspect_ratio: float = 1.3,
        min_extent: float = 0.6,
        approx_epsilon: float = 0.04,
        min_corners: int = 4,
        max_corners: int = 8,
        base_confidence: float = 0.6,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
    ):
        """
        Args:
            blur_kernel: Gaussian blur kernel boyutu
            morph_kernel: Morfolojik işlem kernel boyutu
            min_area: Minimum kontur alanı (piksel²)
            max_area: Maksimum kontur alanı (piksel²)
            min_aspect_ratio: Minimum en/boy oranı (kareye yakınlık)
            max_aspect_ratio: Maksimum en/boy oranı
            min_extent: Minimum extent (kontur alanı / bbox alanı)
            approx_epsilon: Kontur yaklaşıklık toleransı
            min_corners: Minimum köşe sayısı
            max_corners: Maksimum köşe sayısı
            base_confidence: Temel güven skoru
        """
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_extent = min_extent
        self.approx_epsilon = approx_epsilon
        self.min_corners = min_corners
        self.max_corners = max_corners
        self.base_confidence = base_confidence
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

        self.kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    
    def detect(self, image: np.ndarray) -> DetectorOutput:
        """
        Klasik CV ile küp algıla
        
        Args:
            image: BGR görüntü
        
        Returns:
            Algılama sonuçları
        """
        # Ön işleme
        blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Kenar tespiti
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        
        # Morfolojik closing (kenar boşluklarını kapat)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel)
        
        # Kontur bulma
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Küp adaylarını filtrele
        candidates = []
        for contour in contours:
            candidate = self._evaluate_contour(contour)
            if candidate is not None:
                candidates.append(candidate)
        
        # Sonuçları birleştir
        bboxes = []
        classes = []
        confidences = []
        class_names = []
        
        for bbox, confidence in candidates:
            bboxes.append(bbox)
            classes.append(DetectionClass.CUBE)
            confidences.append(confidence)
            class_names.append("cube")
        
        logger.debug(f"Klasik dedektör: {len(candidates)} küp adayı bulundu")
        
        return DetectorOutput(
            bboxes=bboxes,
            classes=classes,
            confidences=confidences,
            class_names=class_names
        )
    
    def _evaluate_contour(
        self,
        contour: np.ndarray
    ) -> Tuple[BoundingBox, float] | None:
        """
        Konturun küp adayı olup olmadığını değerlendir
        
        Args:
            contour: OpenCV kontur
        
        Returns:
            (bbox, confidence) veya None
        """
        # Alan filtresi
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = BoundingBox(x, y, x + w, y + h)
        
        # Aspect ratio (kareye yakınlık)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return None
        
        # Extent (kontur doluluğu)
        extent = area / (w * h) if (w * h) > 0 else 0
        if extent < self.min_extent:
            return None
        
        # Köşe sayısı kontrolü (yaklaşık dörtgen)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_epsilon * perimeter, True)
        num_corners = len(approx)
        
        if num_corners < self.min_corners or num_corners > self.max_corners:
            return None
        
        # Güven skoru hesapla (geometrik özelliklere göre)
        confidence = self._compute_confidence(
            aspect_ratio=aspect_ratio,
            extent=extent,
            num_corners=num_corners,
            area=area
        )
        
        return (bbox, confidence)
    
    def _compute_confidence(
        self,
        aspect_ratio: float,
        extent: float,
        num_corners: int,
        area: float
    ) -> float:
        """
        Geometrik özelliklere göre güven skoru hesapla
        
        Returns:
            0-1 arası güven skoru
        """
        confidence = self.base_confidence
        
        # Kareye ne kadar yakın (aspect_ratio ~ 1.0)
        aspect_score = 1.0 - abs(1.0 - aspect_ratio)
        confidence += 0.1 * aspect_score
        
        # Extent yüksekse (dolu kare)
        extent_score = extent
        confidence += 0.1 * extent_score
        
        # Tam 4 köşe varsa bonus
        if num_corners == 4:
            confidence += 0.1
        
        # Normalize
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return float(confidence)
    
    def get_name(self) -> str:
        return "ClassicalDetector"
    
    @classmethod
    def from_config(cls, config: "DetectionConfig") -> "ClassicalDetector":
        """Konfigürasyondan dedektör oluştur"""
        c = config.classical
        return cls(
            blur_kernel=c.blur_kernel,
            morph_kernel=c.morph_kernel,
            min_area=c.min_area,
            max_area=c.max_area,
            min_aspect_ratio=c.min_aspect_ratio,
            max_aspect_ratio=c.max_aspect_ratio,
            min_extent=c.min_extent,
            approx_epsilon=c.approx_epsilon,
            min_corners=c.min_corners,
            max_corners=c.max_corners,
            base_confidence=c.base_confidence,
            canny_threshold1=c.canny_threshold1,
            canny_threshold2=c.canny_threshold2,
        )
