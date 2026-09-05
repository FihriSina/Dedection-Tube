"""Geometrik hesaplamalar - kaba derinlik tahmini"""
# geometry.py, bounding box'lar ve kamera kalibrasyonu kullanarak nesnelerin konumunu tahmin etmek için kullanılan fonksiyonları içerir. estimate_depth_from_bbox fonksiyonu, bounding box genişliğinden kaba bir derinlik tahmini yaparken, estimate_position_from_bbox fonksiyonu ise bounding box'tan 3D konum tahmini gerçekleştirir. Ayrıca, compute_aspect_score fonksiyonu, bounding box'ın kareye ne kadar yakın olduğunu hesaplayarak bir skor döndürür. Bu fonksiyonlar, pose tahmini pipeline'ında PnP yöntemi başarısız olduğunda kullanılabilecek geometrik bir fallback sağlar.
import numpy as np
import logging
from typing import Optional

from src.utils.types import BoundingBox
from src.calibration.models import CameraCalibration
from src.pose.models import PoseEstimateResult

logger = logging.getLogger(__name__)


def estimate_depth_from_bbox(
    bbox: BoundingBox,
    real_size_m: float,
    camera_calibration: CameraCalibration
) -> Optional[float]:
    """
    Bounding box genişliğinden kaba derinlik tahmini
    
    Z = (f * W_real) / W_pixel
    
    Args:
        bbox: Bounding box
        real_size_m: Küpün gerçek boyutu (metre)
        camera_calibration: Kamera kalibrasyonu
    
    Returns:
        Tahmini derinlik (metre) veya None
    """
    if bbox.width <= 0:
        return None
    
    # Kamera focal length (piksel)
    fx = camera_calibration.camera_matrix[0, 0]
    
    # Derinlik hesabı
    pixel_width = bbox.width
    depth = (fx * real_size_m) / pixel_width
    
    return float(depth)


def estimate_position_from_bbox(
    bbox: BoundingBox,
    real_size_m: float,
    camera_calibration: CameraCalibration
) -> PoseEstimateResult:
    """
    Bounding box'tan basit geometrik konum tahmini
    
    Args:
        bbox: Bounding box
        real_size_m: Küpün gerçek boyutu (metre)
        camera_calibration: Kamera kalibrasyonu
    
    Returns:
        Pose tahmini
    """
    # Derinlik
    z = estimate_depth_from_bbox(bbox, real_size_m, camera_calibration)
    
    if z is None:
        z = 1.0  # Default
        position_confidence = 0.1
    else:
        position_confidence = 0.6  # Orta güven
    
    # Kamera matrisi
    fx = camera_calibration.camera_matrix[0, 0]
    fy = camera_calibration.camera_matrix[1, 1]
    cx = camera_calibration.camera_matrix[0, 2]
    cy = camera_calibration.camera_matrix[1, 2]
    
    # Bbox merkezi
    bbox_center_x, bbox_center_y = bbox.center
    
    # 3D konum (kamera koordinat sisteminde)
    x = (bbox_center_x - cx) * z / fx
    y = (bbox_center_y - cy) * z / fy
    
    # Yönelim (bilinmiyor, varsayılan)
    roll, pitch, yaw = 0.0, 0.0, 0.0
    orientation_confidence = 0.0
    
    return PoseEstimateResult(
        position_x=x,
        position_y=y,
        position_z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        position_confidence=position_confidence,
        orientation_confidence=orientation_confidence,
        method="geometric_bbox"
    )


def compute_aspect_score(bbox: BoundingBox) -> float:
    """
    Bounding box'ın kareye yakınlığını hesapla
    
    Returns:
        0-1 arası skor (1 = tam kare)
    """
    if bbox.height == 0:
        return 0.0
    
    aspect_ratio = bbox.width / bbox.height
    
    # Kareye yakınlık
    score = 1.0 - abs(1.0 - aspect_ratio)
    score = np.clip(score, 0.0, 1.0)
    
    return float(score)
