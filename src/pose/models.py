"""Pose tahmini veri modelleri"""
# models.py, pose tahmini sonuçlarını temsil eden veri modellerini içerir. PoseEstimateResult sınıfı, bir nesnenin 3D pozisyonu, yönelimi ve bu tahminlere ilişkin güven skorlarını içeren bir yapıdır. Ayrıca, ham vektörler (rvec ve tvec) ve kullanılan yöntem bilgisi gibi ek alanlar da bulunur. Bu yapı, pose tahmini sonuçlarını normalize edilmiş bir formatta temsil eder ve pipeline'ın diğer bileşenleri tarafından kullanılmak üzere tasarlanmıştır.
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PoseEstimateResult:
    """Pose tahmini sonucu"""
    
    # 3D pozisyon (metre)
    position_x: float
    position_y: float
    position_z: float  # derinlik
    
    # Yönelim (Euler açıları, derece)
    roll: float
    pitch: float
    yaw: float
    
    # Güven skorları
    position_confidence: float  # 0-1
    orientation_confidence: float  # 0-1
    
    # Ham vektörler
    rvec: Optional[np.ndarray] = None  # Rodrigues rotasyon vektörü
    tvec: Optional[np.ndarray] = None  # Translasyon vektörü
    
    # Yöntem bilgisi
    method: str = "unknown"  # "pnp", "geometric", vb.
    
    def __str__(self) -> str:
        return (
            f"PoseEstimate(\n"
            f"  position: ({self.position_x:.3f}, {self.position_y:.3f}, {self.position_z:.3f}) m\n"
            f"  orientation: (R:{self.roll:.1f}°, P:{self.pitch:.1f}°, Y:{self.yaw:.1f}°)\n"
            f"  confidence: pos={self.position_confidence:.2f}, ori={self.orientation_confidence:.2f}\n"
            f"  method: {self.method}\n"
            f")"
        )
