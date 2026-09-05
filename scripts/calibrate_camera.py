"""Kamera kalibrasyon scripti - checkerboard kullanarak"""

import cv2
import numpy as np
import argparse # komut satırı argümanlarını işlemek için
import logging # loglama için
from pathlib import Path # dosya yollarını yönetmek için
from typing import List, Tuple, Optional # tip ipuçları için
import sys

# Proje kökünden import için path ekle

from src.utils.config import build_configs
from src.utils.logging import setup_logger
from src.camera.opencv_camera import OpenCVCamera
from src.calibration.models import CameraCalibration
from src.calibration.io import save_calibration


def calibrate_camera(
    images: List[np.ndarray],
    checkerboard_size: Tuple[int, int],
    square_size_mm: float
) -> Optional[CameraCalibration]:
    """
    Checkerboard görüntülerinden kamera kalibrasyonu yap
    
    Args:
        images: Kalibrasyon görüntüleri
        checkerboard_size: İç köşe sayısı (width, height)
        square_size_mm: Kare boyutu (mm)
    
    Returns:
        Kalibrasyon verisi veya None
    """
    logger = logging.getLogger(__name__)
    
    # 3D nokta koordinatları (checkerboard düzlemi)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32) # Checkerboard üzerindeki her köşe için 3D koordinatları tutacak bir dizi oluşturur. Bu dizi, checker 
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) # Checkerboard üzerindeki köşe noktalarının 2D koordinatlarını oluşturur. np.mgrid fonksiyonu, belirtilen aralıkta bir ızgara oluşturur ve bu ızgara, checkerboard üzerindeki köşe noktalarının konumlarını temsil eder. Bu koordinatlar daha sonra 3D koordinatlara dönüştürülür.
    objp *= square_size_mm  # mm cinsinden
    
    # 3D ve 2D nokta koleksiyonları
    objpoints = []  # 3D dünya koordinatları
    imgpoints = []  # 2D görüntü koordinatları
    
    image_height, image_width = images[0].shape[:2]
    
    logger.info(f"Toplam {len(images)} görüntü işleniyor...")
    
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Checkerboard köşelerini bul
        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Sub-pixel hassasiyetinde köşe konumları
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            logger.info(f"Görüntü {idx + 1}: Checkerboard bulundu ✓")
        else:
            logger.warning(f"Görüntü {idx + 1}: Checkerboard bulunamadı ✗")
    
    if len(objpoints) < 3: # Kalibrasyon için en az 3 farklı görüntü: 
        logger.error(f"Yetersiz geçerli görüntü: {len(objpoints)} < 3")
        return None
    
    logger.info(f"{len(objpoints)} görüntü kullanılarak kalibrasyon yapılıyor...")
    
    # Kalibrasyon hesapla
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (image_width, image_height),
        None,
        None
    )
    
    if not ret: # başarısızlık kontrolü
        logger.error("Kalibrasyon başarısız")
        return None
    
    # Reprojection error hesapla
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    logger.info(f"Kalibrasyon başarılı!")
    logger.info(f"RMS reprojection error: {mean_error:.4f} piksel")
    #Kalibrasyon verlerini nesneye paketliyor
    calibration = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_width,
        image_height=image_height,
        reprojection_error=mean_error,
        num_images=len(objpoints)
    )
    
    return calibration


def collect_calibration_images(
    camera: OpenCVCamera,
    min_images: int = 15,
    max_images: int = 30,
    checkerboard_size: Tuple[int, int] = (9, 6)
) -> List[np.ndarray]:
    """
    Canlı kameradan kalibrasyon görüntüleri topla
    
    Args:
        camera: Kamera nesnesi
        min_images: Minimum görüntü sayısı
        max_images: Maksimum görüntü sayısı
        checkerboard_size: Checkerboard boyutu
    
    Returns:
        Toplanan görüntüler
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Kalibrasyon görüntüsü toplama modu")
    logger.info(f"Hedef: {min_images}-{max_images} görüntü")
    logger.info("Checkerboard'u farklı açılardan gösterin")
    logger.info("Tuşlar: [SPACE] = yakala, [Q] = bitir")
    
    images = []
    
    while len(images) < max_images:
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error("Frame okunamadı")
            break
        
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Checkerboard tespiti (görsel feedback için)
        ret_corners, corners = cv2.findChessboardCorners(
            gray,
            checkerboard_size,
            cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret_corners:
            cv2.drawChessboardCorners(display, checkerboard_size, corners, ret_corners)
            status_text = "Checkerboard BULUNDU - SPACE ile yakala"
            color = (0, 255, 0)
        else:
            status_text = "Checkerboard aranıyor..."
            color = (0, 0, 255)
        
        # Bilgi overlay
        cv2.putText(
            display,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        cv2.putText(
            display,
            f"Toplanan: {len(images)}/{max_images} (min: {min_images})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Kalibrasyon - Görüntü Toplama", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space
            if ret_corners:
                images.append(frame.copy())
                logger.info(f"Görüntü {len(images)} kaydedildi ✓")
            else:
                logger.warning("Checkerboard bulunamadı, görüntü atlandı")
        
        elif key == ord('q'):  # Quit
            if len(images) >= min_images:
                logger.info(f"Kullanıcı durdurdu, {len(images)} görüntü toplandı")
                break
            else:
                logger.warning(f"En az {min_images} görüntü gerekli (şu an: {len(images)})")
    
    cv2.destroyAllWindows()
    return images


def main():
    parser = argparse.ArgumentParser(description="Kamera kalibrasyon scripti")
    parser.add_argument(
        "--output",
        type=str,
        default="camera_calibration.npz",
        help="Çıktı dosyası yolu"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi"
    )

    args = parser.parse_args()
    
    # Logger kur
    logger = setup_logger("calibration", level=args.log_level)
    
    # Config yükle
    try:
        configs = build_configs()
        calib_config = configs.calibration
    except Exception as e:
        logger.error(f"Config yüklenemedi: {e}")
        return 1

    # Checkerboard parametreleri
    checkerboard_size = tuple(calib_config.checkerboard_size)
    square_size_mm = calib_config.square_size_mm
    min_images = calib_config.min_images
    max_images = calib_config.max_images

    logger.info(f"Checkerboard: {checkerboard_size[0]}x{checkerboard_size[1]}")
    logger.info(f"Kare boyutu: {square_size_mm} mm")

    # Kamera aç
    camera = OpenCVCamera.from_config(configs.camera)
    if not camera.open():
        logger.error("Kamera açılamadı")
        return 1
    
    try:
        # Görüntü topla
        images = collect_calibration_images(
            camera,
            min_images=min_images,
            max_images=max_images,
            checkerboard_size=checkerboard_size
        )
        
        if len(images) < min_images:
            logger.error(f"Yetersiz görüntü: {len(images)} < {min_images}")
            return 1
        
        # Kalibrasyon yap
        calibration = calibrate_camera(
            images,
            checkerboard_size,
            square_size_mm
        )
        
        if calibration is None:
            logger.error("Kalibrasyon başarısız")
            return 1
        
        # Kaydet
        output_path = Path(args.output)
        if save_calibration(calibration, output_path):
            logger.info(f"✓ Kalibrasyon başarıyla kaydedildi: {output_path}")
            logger.info(f"\n{calibration}")
            return 0
        else:
            logger.error("Kalibrasyon kaydedilemedi")
            return 1
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
