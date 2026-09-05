"""Algılama veri modelleri"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.utils.types import BoundingBox, DetectionClass


@dataclass
class DetectorOutput:
    """Dedektör çıktısı - normalize edilmiş format.

    track_ids alanı BoxMOT gibi tracker katmanları tarafından doldurulur ve
    bboxes/classes/confidences/class_names listeleriyle aynı indeks hizasında tutulur.
    Tracking kapalıysa boş kalabilir; bu durumda sonuç birleştirici frame içi indeks kullanır.
    """

    bboxes: List[BoundingBox]
    classes: List[DetectionClass]
    confidences: List[float]
    class_names: List[str]
    track_ids: List[Optional[int]] = field(default_factory=list)

    def __len__(self) -> int:
        """Dedektör çıktısındaki tespit sayısını döndürür."""
        return len(self.bboxes)

    def is_empty(self) -> bool:
        return len(self) == 0

    def get_track_id(self, index: int) -> Optional[int]:
        """Verilen tespit indeksine ait takip ID'sini güvenli şekilde döndür."""
        if 0 <= index < len(self.track_ids):
            return self.track_ids[index]
        return None

    def with_track_ids(self, track_ids: List[Optional[int]]) -> "DetectorOutput":
        """Aynı tespitleri yeni takip ID listesiyle kopyala."""
        return DetectorOutput(
            bboxes=self.bboxes.copy(),
            classes=self.classes.copy(),
            confidences=self.confidences.copy(),
            class_names=self.class_names.copy(),
            track_ids=track_ids.copy(),
        )
