"""Kalibrasyon veri modelleri"""
# models.py, kamera kalibrasyon verilerini temsil eden veri modellerini içerir. CameraCalibration sınıfı, bir kameranın intrinsic parametrelerini (camera matrix), distorsiyon katsayılarını, görüntü çözünürlüğünü ve reprojection error gibi bilgileri içeren bir yapıdır. Ayrıca, kalibrasyonun geçerli olup olmadığını kontrol etmek için is_valid yöntemi ve okunabilir bir string formatı için __str__ yöntemi de sağlanır. Bu yapı, kalibrasyon verilerini normalize edilmiş bir formatta temsil eder ve pipeline'ın diğer bileşenleri tarafından kullanılmak üzere tasarlanmıştır.
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CameraCalibration:
    """Kamera kalibrasyon verileri"""
    
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray    # Distorsiyon katsayıları (k1, k2, p1, p2, k3, ...)
    image_width: int
    image_height: int
    reprojection_error: float  # RMS reprojection error (piksel)
    
    # Opsiyonel ek bilgi
    calibration_date: Optional[str] = None
    num_images: Optional[int] = None
    
    def is_valid(self, max_error: float = 0.5) -> bool:
        """
        Kalibrasyonun geçerli olup olmadığını kontrol et
        
        Args:
            max_error: Maksimum kabul edilebilir reprojection error (piksel)
        
        Returns:
            Geçerli ise True
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False
        
        if self.camera_matrix.shape != (3, 3):
            return False
        
        if self.reprojection_error > max_error:
            return False
        
        return True
    
    def __str__(self) -> str:
        """Okunabilir string formatı"""
        return (
            f"CameraCalibration(\n"
            f"  resolution: {self.image_width}x{self.image_height}\n"
            f"  fx: {self.camera_matrix[0, 0]:.2f}\n"
            f"  fy: {self.camera_matrix[1, 1]:.2f}\n"
            f"  cx: {self.camera_matrix[0, 2]:.2f}\n"
            f"  cy: {self.camera_matrix[1, 2]:.2f}\n"
            f"  reprojection_error: {self.reprojection_error:.4f} px\n"
            f"  num_images: {self.num_images}\n"
            f")"
        )
