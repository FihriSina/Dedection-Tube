"""Kalibrasyon I/O işlemleri"""
# io.py, kamera kalibrasyon verilerini .npz dosyalarına kaydetmek ve bu dosyalardan yüklemek için kullanılan fonksiyonları içerir. save_calibration fonksiyonu, CameraCalibration nesnesini verilen bir dosya yoluna kaydederken, load_calibration fonksiyonu belirtilen dosya yolundan kalibrasyon verilerini yükler ve bir CameraCalibration nesnesi olarak döndürür. Her iki fonksiyon da işlemin başarılı olup olmadığını belirten boolean değerler döndürür ve hata durumlarında uygun log mesajları üretir. Bu modül, kalibrasyon verilerinin yönetimi için merkezi bir nokta sağlar ve pipeline'ın diğer bileşenleri tarafından kullanılmak üzere tasarlanmıştır.
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.calibration.models import CameraCalibration

logger = logging.getLogger(__name__)


def save_calibration(
    calibration: CameraCalibration,
    file_path: Path
) -> bool:
    """
    Kalibrasyon verilerini .npz dosyasına kaydet
    
    Args:
        calibration: Kalibrasyon verisi
        file_path: Kayıt yolu
    
    Returns:
        Başarılı ise True
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            file_path,
            camera_matrix=calibration.camera_matrix,
            dist_coeffs=calibration.dist_coeffs,
            image_width=calibration.image_width,
            image_height=calibration.image_height,
            reprojection_error=calibration.reprojection_error,
            calibration_date=calibration.calibration_date or datetime.now().isoformat(),
            num_images=calibration.num_images or 0
        )
        
        logger.info(f"Kalibrasyon kaydedildi: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Kalibrasyon kaydedilemedi: {e}")
        return False


def load_calibration(file_path: Path) -> Optional[CameraCalibration]:
    """
    Kalibrasyon verilerini .npz dosyasından yükle
    
    Args:
        file_path: Kalibrasyon dosyası yolu
    
    Returns:
        Kalibrasyon verisi veya None
    """
    if not file_path.exists():
        logger.error(f"Kalibrasyon dosyası bulunamadı: {file_path}")
        return None
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        calibration = CameraCalibration(
            camera_matrix=data["camera_matrix"],
            dist_coeffs=data["dist_coeffs"],
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            reprojection_error=float(data["reprojection_error"]),
            calibration_date=str(data.get("calibration_date", "")),
            num_images=int(data.get("num_images", 0))
        )
        
        logger.info(f"Kalibrasyon yüklendi: {file_path}")
        logger.debug(f"\n{calibration}")
        
        return calibration
        
    except Exception as e:
        logger.error(f"Kalibrasyon yüklenemedi: {e}")
        return None
