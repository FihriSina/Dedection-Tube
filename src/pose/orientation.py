"""Yönelim tahmini yardımcı fonksiyonları"""
# orientation.py, bounding box'lar ve görüntü boyutları kullanarak nesnelerin yönelimini tahmin etmek için kullanılan fonksiyonları içerir. estimate_orientation_from_bbox fonksiyonu, bounding box'ın görüntü merkezine göre konumuna dayanarak basit bir roll, pitch ve yaw tahmini yapar. refine_orientation_with_aspect fonksiyonu ise bounding box'ın aspect ratio'sunu kullanarak yönelim tahminini iyileştirmeye çalışır. Bu fonksiyonlar, pose tahmini pipeline'ında PnP yöntemi başarısız olduğunda kullanılabilecek geometrik bir fallback sağlar.
import numpy as np
from typing import Optional

from src.utils.types import BoundingBox
from src.pose.models import PoseEstimateResult


def estimate_orientation_from_bbox(
    bbox: BoundingBox,
    image_shape: tuple[int, int]
) -> tuple[float, float, float]:
    """
    Bbox pozisyonundan basit yönelim tahmini
    
    Kamera merkezi referans alınır; bbox merkezi kameranın
    sağında/solunda/yukarı/aşağıda ise yaw/pitch tahmin edilir
    
    Args:
        bbox: Bounding box
        image_shape: (height, width)
    
    Returns:
        (roll, pitch, yaw) derece
    """
    image_height, image_width = image_shape
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    bbox_center_x, bbox_center_y = bbox.center
    
    # Yatay sapma -> yaw
    horizontal_offset = (bbox_center_x - image_center_x) / image_width
    yaw = horizontal_offset * 30.0  # Maksimum ±30 derece
    
    # Dikey sapma -> pitch
    vertical_offset = (bbox_center_y - image_center_y) / image_height
    pitch = -vertical_offset * 20.0  # Maksimum ±20 derece (ters yön)
    
    # Roll (bilinmiyor)
    roll = 0.0
    
    return float(roll), float(pitch), float(yaw)


def refine_orientation_with_aspect(
    pose: PoseEstimateResult,
    bbox: BoundingBox
) -> PoseEstimateResult:
    """
    Bbox aspect ratio ile yönelim iyileştirmesi
    
    Args:
        pose: Mevcut pose tahmini
        bbox: Bounding box
    
    Returns:
        İyileştirilmiş pose
    """
    aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0
    
    # Aspect ratio 1'den uzaklaştıkça pitch/yaw varyansı artar
    deviation = abs(1.0 - aspect_ratio)
    
    # Güven skorunu düşür (geometri bozulmuş)
    if deviation > 0.3:
        pose.orientation_confidence *= 0.7
    
    return pose
