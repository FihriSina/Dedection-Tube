"""solvePnP tabanlı pose tahmini"""
# Pnp POSE TAHMİNİ
import cv2
import numpy as np
import logging
from typing import Optional

from src.calibration.models import CameraCalibration
from src.utils.types import BoundingBox
from src.utils.config_models import PnPConfig, CornersConfig
from src.pose.models import PoseEstimateResult
from src.pose.corners import estimate_cube_face_corners

logger = logging.getLogger(__name__)

_DEFAULT_PNP_CONFIG = PnPConfig()


def estimate_pose_pnp(
    image: np.ndarray,
    bbox: BoundingBox,
    cube_size_m: float,
    camera_calibration: CameraCalibration,
    pnp_config: Optional[PnPConfig] = None,
    corners_config: Optional[CornersConfig] = None,
) -> Optional[PoseEstimateResult]:
    """
    solvePnP ile küp pose tahmini

    Args:
        image: Gri veya BGR görüntü
        bbox: Küp bounding box
        cube_size_m: Küpün gerçek boyutu (metre)
        camera_calibration: Kamera kalibrasyonu
        pnp_config: PnP parametreleri (None ise varsayılan kullanılır)
        corners_config: Köşe tespiti parametreleri (None ise varsayılan kullanılır)

    Returns:
        Pose tahmini veya None
    """
    cfg = pnp_config or _DEFAULT_PNP_CONFIG

    # 2D köşe noktaları tespit et
    image_points = estimate_cube_face_corners(image, bbox, config=corners_config)

    if image_points is None or len(image_points) != 4:
        logger.debug("solvePnP için yetersiz köşe noktası")
        return None

    # 3D model noktaları (küp yüzü, merkez orijinde)
    half_size = cube_size_m / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],  # Sol üst
        [ half_size, -half_size, 0],  # Sağ üst
        [ half_size,  half_size, 0],  # Sağ alt
        [-half_size,  half_size, 0],  # Sol alt
    ], dtype=np.float32)

    # Kamera matrisi ve distorsiyon katsayıları
    camera_matrix = camera_calibration.camera_matrix
    dist_coeffs = camera_calibration.dist_coeffs

    # solvePnP
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            logger.debug("solvePnP başarısız")
            return None

        # Translasyon vektöründen pozisyon (metre)
        x, y, z = tvec.flatten()

        # Rodrigues vektöründen rotasyon matrisi
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Euler açıları hesapla
        roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)

        # Derinlik makul aralıkta mı
        if cfg.min_depth_m < z < cfg.max_depth_m:
            position_confidence = cfg.high_confidence
        else:
            position_confidence = cfg.low_confidence

        # Yönelim güveni (4 köşe tespit edildiyse yüksek)
        orientation_confidence = cfg.orientation_confidence

        result = PoseEstimateResult(
            position_x=x,
            position_y=y,
            position_z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            position_confidence=position_confidence,
            orientation_confidence=orientation_confidence,
            rvec=rvec,
            tvec=tvec,
            method="solvepnp"
        )

        logger.debug(f"solvePnP başarılı: z={z:.3f}m")
        return result

    except Exception as e:
        logger.error(f"solvePnP hatası: {e}")
        return None


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    Rotasyon matrisinden Euler açıları (derece)

    Args:
        R: 3x3 rotasyon matrisi

    Returns:
        (roll, pitch, yaw) derece cinsinden
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))
