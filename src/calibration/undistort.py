"""Lens distorsiyon düzeltme"""
# undistort.py, kamera kalibrasyon verilerini kullanarak görüntülerdeki lens distorsiyonunu düzeltmek için kullanılan bir modül içerir. Undistorter sınıfı, verilen bir CameraCalibration nesnesi ile başlatılır ve bu kalibrasyon bilgilerini kullanarak hızlı bir şekilde distorsiyon düzeltmesi yapar. Undistort yöntemi, giriş görüntüsünü düzeltilmiş bir versiyonunu döndürürken, get_new_camera_matrix ve get_roi yöntemleri de düzeltilmiş görüntü için yeni kamera matrisi ve geçerli piksel ROI'sini sağlar. Bu modül, pipeline'ın kalibrasyon ve görüntü işleme aşamalarında önemli bir rol oynar ve diğer bileşenler tarafından kullanılmak üzere tasarlanmıştır.
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

from src.calibration.models import CameraCalibration

logger = logging.getLogger(__name__)


class Undistorter:
    """Lens distorsiyon düzeltme sınıfı"""
    
    def __init__(self, calibration: CameraCalibration):
        """
        Args:
            calibration: Kamera kalibrasyon verisi
        """
        self.calibration = calibration
        
        # Optimal yeni kamera matrisi ve ROI hesapla
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            calibration.camera_matrix,
            calibration.dist_coeffs,
            (calibration.image_width, calibration.image_height),
            alpha=1,  # 1 = tüm pikselleri koru, 0 = sadece geçerli pikseller
            newImgSize=(calibration.image_width, calibration.image_height)
        )
        
        # Undistortion haritaları oluştur (hızlı işlem için)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.dist_coeffs,
            None,
            self.new_camera_matrix,
            (calibration.image_width, calibration.image_height),
            cv2.CV_32FC1
        )
        
        logger.debug("Undistortion haritaları oluşturuldu")
    
    def undistort(
        self,
        image: np.ndarray,
        crop_to_roi: bool = False
    ) -> np.ndarray:
        """
        Görüntünün distorsiyon düzeltmesi
        
        Args:
            image: Giriş görüntüsü
            crop_to_roi: ROI'ye kırp
        
        Returns:
            Düzeltilmiş görüntü
        """
        # Harita kullanarak hızlı remap
        undistorted = cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR
        )
        
        # ROI'ye kırp (isteğe bağlı)
        if crop_to_roi:
            x, y, w, h = self.roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_new_camera_matrix(self) -> np.ndarray:
        """Düzeltilmiş görüntü için kamera matrisi"""
        return self.new_camera_matrix
    
    def get_roi(self) -> Tuple[int, int, int, int]:
        """Geçerli piksel ROI (x, y, width, height)"""
        return self.roi
