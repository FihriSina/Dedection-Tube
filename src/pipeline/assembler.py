"""Ana sonuç birleştirici - tüm modülleri bir araya getirir"""

import logging
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from src.utils.types import (
    CubeDetectionResult, FrameAnalysisResult,
    Detection, DetectionClass
)
from src.detection.models import DetectorOutput
from src.calibration.models import CameraCalibration
from src.pose.pnp import estimate_pose_pnp
from src.pose.geometry import estimate_position_from_bbox

logger = logging.getLogger(__name__)

# ResultAssembler, dedektörün çıktısını işleyerek küp sonuçları oluşturur.
# PnP (Perspective-n-Point) algoritmasıyla küp poz tahmini yapar, ancak başarısız
# olursa geometrik yöntemlere geri döner.
class ResultAssembler:
    """Tüm modül sonuçlarını birleştir"""

    def __init__(
        self,
        cube_size_m: float = 0.09,
        use_pnp: bool = True,
        pnp_fallback_to_geometric: bool = True,
    ):
        """
        Args:
            cube_size_m: Küp boyutu (metre)
            use_pnp: solvePnP kullan
            pnp_fallback_to_geometric: PnP başarısızsa geometrik fallback
        """
        self.cube_size_m = cube_size_m
        self.use_pnp = use_pnp
        self.pnp_fallback_to_geometric = pnp_fallback_to_geometric

    def assemble(
        self,
        image: np.ndarray,
        detections: DetectorOutput,
        calibration: Optional[CameraCalibration],
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> FrameAnalysisResult:
        """
        Frame analiz sonucunu oluştur

        Args:
            image: BGR görüntü
            detections: Algılama sonuçları
            calibration: Kamera kalibrasyonu
            frame_id: Frame numarası
            timestamp: Zaman damgası (saniye)

        Returns:
            Tam frame analiz sonucu
        """
        cubes: List[CubeDetectionResult] = []

        # Küp algılamalarını işle
        for i in range(len(detections)):
            if detections.classes[i] != DetectionClass.CUBE:
                continue

            bbox = detections.bboxes[i]
            confidence = detections.confidences[i]
            class_name = detections.class_names[i]

            detection = Detection(
                bbox=bbox,
                class_id=DetectionClass.CUBE,
                confidence=confidence,
                class_name=class_name,
                track_id=detections.get_track_id(i),
            )

            # Pose tahmini
            pose = None
            if calibration is not None:
                # Önce solvePnP dene
                if self.use_pnp:
                    pose_result = estimate_pose_pnp(
                        image, bbox, self.cube_size_m, calibration
                    )

                    if pose_result is not None:
                        from src.utils.types import Position3D, Orientation, CubePose

                        position = Position3D(
                            x=pose_result.position_x,
                            y=pose_result.position_y,
                            z=pose_result.position_z,
                            confidence=pose_result.position_confidence
                        )

                        orientation = Orientation(
                            roll=pose_result.roll,
                            pitch=pose_result.pitch,
                            yaw=pose_result.yaw,
                            confidence=pose_result.orientation_confidence
                        )

                        pose = CubePose(
                            position=position,
                            orientation=orientation,
                            rvec=pose_result.rvec,
                            tvec=pose_result.tvec
                        )

                # PnP başarısızsa geometrik fallback
                if pose is None and self.pnp_fallback_to_geometric:
                    geom_result = estimate_position_from_bbox(
                        bbox, self.cube_size_m, calibration
                    )

                    from src.utils.types import Position3D, Orientation, CubePose

                    position = Position3D(
                        x=geom_result.position_x,
                        y=geom_result.position_y,
                        z=geom_result.position_z,
                        confidence=geom_result.position_confidence
                    )

                    orientation = Orientation(
                        roll=geom_result.roll,
                        pitch=geom_result.pitch,
                        yaw=geom_result.yaw,
                        confidence=geom_result.orientation_confidence
                    )

                    pose = CubePose(
                        position=position,
                        orientation=orientation
                    )

            cube_result = CubeDetectionResult(
                detection=detection,
                pose=pose
            )

            cubes.append(cube_result)

        result = FrameAnalysisResult(
            cubes=cubes,
            timestamp=timestamp,
            frame_id=frame_id,
        )

        logger.debug(f"Frame {frame_id}: {len(cubes)} küp")

        return result
