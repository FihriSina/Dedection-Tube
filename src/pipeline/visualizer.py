"""Görselleştirme yardımcıları"""

import dataclasses
import cv2
import numpy as np
from typing import Optional, Dict, Any

from src.utils.types import FrameAnalysisResult


class Visualizer:
    """Sonuçları görüntü üzerine çiz"""

    def __init__(
        self,
        draw_bbox: bool = True,
        draw_label: bool = True,
        draw_confidence: bool = True,
        draw_center: bool = True,
        draw_axes: bool = False,
        colors: Optional[Dict[str, tuple]] = None,
        font_scale: float = 0.3,
        font_thickness: int = 1,
        bbox_alpha: float = 0.3
    ):
        """
        Args:
            draw_bbox: Bbox çiz
            draw_label: Etiket çiz
            draw_confidence: Güven skoru çiz
            draw_center: Merkez nokta çiz
            draw_axes: 3D eksen çiz (pose varsa)
            colors: Renk tanımları (BGR)
            font_scale: Font ölçeği
            font_thickness: Font kalınlığı
            bbox_alpha: Bbox opacity
        """
        self.draw_bbox = draw_bbox
        self.draw_label = draw_label
        self.draw_confidence = draw_confidence
        self.draw_center = draw_center
        self.draw_axes = draw_axes
        self.colors = colors or {
            "cube": (0, 255, 0),
            "text": (255, 255, 255)
        }
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.bbox_alpha = bbox_alpha

    def draw_results(
        self,
        image: np.ndarray,
        results: FrameAnalysisResult
    ) -> np.ndarray:
        """
        Sonuçları görüntü üzerine çiz
        
        Args:
            image: BGR görüntü
            results: Frame analiz sonuçları
        
        Returns:
            Çizilmiş görüntü
        """
        output = image.copy()
        
        # Küpleri çiz
        for cube in results.cubes:
            self._draw_cube(output, cube)

        self._draw_info(output, results)

        return output

    def _draw_cube(self, image: np.ndarray, cube) -> None:
        """Küp çiz"""
        bbox = cube.detection.bbox
        color = self.colors["cube"]
        
        # Bbox
        if self.draw_bbox:
            cv2.rectangle(
                image,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                2
            )
        
        # Merkez
        if self.draw_center:
            center = bbox.center
            cv2.circle(image, center, 5, color, -1)
        
        # Etiket
        if self.draw_label or self.draw_confidence:
            label_parts = []

            if self.draw_label:
                label_parts.append("CUBE")
                if getattr(cube.detection, "track_id", None) is not None:
                    label_parts.append(f"ID:{cube.detection.track_id}")

            if self.draw_confidence:
                label_parts.append(f"{cube.detection.confidence:.2f}")

            label = " ".join(label_parts)
            
            # Konum bilgisi varsa ekle
            if cube.pose is not None:
                z = cube.pose.position.z
                label += f" | Z:{z:.2f}m"
            
            # Metin arkaplanı
            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.font_thickness
            )

            cv2.rectangle(
                image,
                (bbox.x1, bbox.y1 - h - 10),
                (bbox.x1 + w + 10, bbox.y1),
                color,
                -1
            )
            
            # Metin
            cv2.putText(
                image,
                label,
                (bbox.x1 + 5, bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.colors["text"],
                self.font_thickness
            )

    def _draw_info(self, image: np.ndarray, results: FrameAnalysisResult) -> None:
        """Genel bilgi çiz"""
        info_text = f"Frame: {results.frame_id} | Kupler: {len(results.cubes)}"

        cv2.putText(
            image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    @classmethod
    def from_config(cls, config: "DemoConfig") -> "Visualizer":
        """Konfigürasyondan visualizer oluştur"""
        v = config.visualization
        return cls(
            draw_bbox=v.draw_bbox,
            draw_label=v.draw_label,
            draw_confidence=v.draw_confidence,
            draw_center=v.draw_center,
            draw_axes=v.draw_axes,
            colors=v.colors.model_dump(),
            font_scale=v.font_scale,
            font_thickness=v.font_thickness,
            bbox_alpha=v.bbox_alpha,
        )
