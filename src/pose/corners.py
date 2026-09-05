"""Küp köşe tespiti"""
# corners.py, görüntülerdeki küp tespiti için köşe tespiti ve geometrik hesaplamalar içeren bir modüldür.
# detect_cube_corners fonksiyonu, verilen bir görüntü ve bounding box içinde küpün köşelerini tespit eder.
# estimate_cube_face_corners fonksiyonu ise daha spesifik olarak küpün yüzeyindeki köşeleri tahmin eder.
# sort_corners_clockwise fonksiyonu, tespit edilen köşeleri saat yönünde sıralar. Bu modül, pose tahmini
# pipeline'ında PnP yöntemi başarısız olduğunda kullanılabilecek geometrik bir fallback sağlar.
import logging
from typing import Optional

import cv2
import numpy as np

from src.utils.types import BoundingBox
from src.utils.config_models import CornersConfig

logger = logging.getLogger(__name__)

_DEFAULT_CORNERS_CONFIG = CornersConfig()

# _extract_roi, verilen bir görüntü ve bounding box'a göre, bounding box çevresinde belirli bir padding ile
# bir ROI (Region of Interest) çıkarır. Bu, köşe tespiti işlemi için daha küçük ve odaklanmış bir alan sağlar.
def _extract_roi(
    image: np.ndarray,
    bbox: BoundingBox,
    padding: int = 6,
) -> tuple[np.ndarray, int, int]:
    image_height, image_width = image.shape[:2]
    x1 = max(0, bbox.x1 - padding)
    y1 = max(0, bbox.y1 - padding)
    x2 = min(image_width, bbox.x2 + padding)
    y2 = min(image_height, bbox.y2 + padding)
    return image[y1:y2, x1:x2], x1, y1

# _sort_points_clockwise, verilen bir dizi köşe noktasını saat yönünde sıralar.
# Bu, köşe noktalarının belirli bir sırayla (örneğin, sol üst, sağ üst, sağ alt, sol alt) düzenlenmesini sağlar.
# Sıralama, köşe noktalarının merkezine göre açılar hesaplanarak gerçekleştirilir.
def _sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    top_left_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -top_left_idx, axis=0).astype(np.float32)


def detect_cube_corners(
    image: np.ndarray,
    bbox: BoundingBox,
    method: str = "goodfeatures",
    config: Optional[CornersConfig] = None,
) -> Optional[np.ndarray]:
    cfg = config or _DEFAULT_CORNERS_CONFIG
    roi, offset_x, offset_y = _extract_roi(image, bbox, padding=cfg.roi_padding)
    if roi.size == 0:
        return None

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    if method == "goodfeatures":
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=cfg.max_corners,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance,
            blockSize=cfg.block_size,
        )
    elif method == "harris":
        gray_float = np.float32(gray)
        response = cv2.cornerHarris(
            gray_float,
            blockSize=cfg.harris_block_size,
            ksize=cfg.harris_ksize,
            k=cfg.harris_k,
        )
        response = cv2.dilate(response, None)
        threshold = cfg.harris_threshold * response.max()
        y_coords, x_coords = np.where(response > threshold)
        if len(x_coords) == 0:
            return None
        corners = np.column_stack([x_coords, y_coords]).astype(np.float32).reshape(-1, 1, 2)
    else:
        logger.warning("Bilinmeyen köşe tespit yöntemi: %s", method)
        return None

    if corners is None or len(corners) == 0:
        return None

    corners = corners.astype(np.float32)
    corners[:, :, 0] += offset_x
    corners[:, :, 1] += offset_y
    return corners

# estimate_cube_face_corners, verilen bir görüntü ve bounding box içinde küpün yüzeyindeki köşeleri tahmin eder.
# İlk olarak, goodFeaturesToTrack ve Harris köşe tespiti yöntemlerini deneyerek köşe noktalarını bulmaya çalışır.
# Eğer bu yöntemler yeterli köşe tespiti yapamazsa, Canny kenar tespiti ve kontur analizi kullanarak daha kaba
# bir köşe tahmini yapar. Son olarak, bulunan köşeler saat yönünde sıralanır ve döndürülür.
# Eğer hiçbir yöntemle yeterli köşe bulunamazsa, bounding box'ın köşeleri varsayılan olarak kullanılır.
def estimate_cube_face_corners(
    image: np.ndarray,
    bbox: BoundingBox,
    config: Optional[CornersConfig] = None,
) -> Optional[np.ndarray]:
    cfg = config or _DEFAULT_CORNERS_CONFIG
    roi, offset_x, offset_y = _extract_roi(image, bbox, padding=cfg.roi_padding)
    if roi.size == 0:
        return None

    roi_bbox = BoundingBox(0, 0, roi.shape[1], roi.shape[0])

    for method in ("goodfeatures", "harris"):
        detected = detect_cube_corners(roi, roi_bbox, method=method, config=cfg)
        if detected is None or len(detected) < 4:
            continue

        points = detected.reshape(-1, 2).astype(np.float32)
        hull = cv2.convexHull(points).reshape(-1, 2)
        candidate = hull

        if len(candidate) > 4:
            perimeter = cv2.arcLength(candidate.reshape(-1, 1, 2), True)
            approx = cv2.approxPolyDP(candidate.reshape(-1, 1, 2), cfg.approx_epsilon * perimeter, True)
            candidate = approx.reshape(-1, 2).astype(np.float32)

        if len(candidate) >= 4:
            if len(candidate) > 4:
                rect = cv2.minAreaRect(candidate)
                candidate = cv2.boxPoints(rect)
            candidate[:, 0] += offset_x
            candidate[:, 1] += offset_y
            return _sort_points_clockwise(candidate)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # Kenar tespiti (köşe yöntemi başarısız olursa kontur tabanlı fallback)
    edges = cv2.Canny(gray, cfg.canny_threshold1, cfg.canny_threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, cfg.approx_epsilon * perimeter, True)
        candidate = approx.reshape(-1, 2).astype(np.float32)
        if len(candidate) >= 4:
            if len(candidate) > 4:
                rect = cv2.minAreaRect(candidate)
                candidate = cv2.boxPoints(rect)
            candidate[:, 0] += offset_x
            candidate[:, 1] += offset_y
            return _sort_points_clockwise(candidate)

    logger.debug("Dörtgen bulunamadı, bbox köşeleri kullanılıyor")
    return np.array(
        [
            [bbox.x1, bbox.y1],
            [bbox.x2, bbox.y1],
            [bbox.x2, bbox.y2],
            [bbox.x1, bbox.y2],
        ],
        dtype=np.float32,
    )


def sort_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    return _sort_points_clockwise(corners)
