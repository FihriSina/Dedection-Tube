"""OpenCV tabanlı kamera implementasyonu"""
# opencv_camera.py, OpenCV'nin VideoCapture sınıfını kullanarak kamera erişimi sağlayan bir modül içerir. OpenCVCamera sınıfı, CameraInterface'i implement eder ve kamerayı açma, frame okuma, serbest bırakma ve özellik ayarlama gibi işlemleri gerçekleştirir. Ayrıca, yeniden bağlanma mekanizması da içerir; eğer frame okuma başarısız olursa, belirli sayıda deneme ile kamerayı yeniden açmaya çalışır. Konfigürasyondan kamera oluşturmak için from_config sınıf yöntemi de sağlanır. Bu modül, pipeline'ın kamera erişimi için temel bir bileşen olarak kullanılabilir.
import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

from src.camera.interface import CameraInterface

logger = logging.getLogger(__name__)


class OpenCVCamera(CameraInterface):
    """OpenCV VideoCapture sarmalayıcı"""
    
    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        auto_exposure: bool = False,
        exposure: Optional[float] = None,
        auto_white_balance: bool = False,
        white_balance: Optional[int] = None,
        buffer_size: int = 1,
        reconnect_enabled: bool = True,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0
    ):
        """
        Args:
            device_index: Kamera cihaz numarası
            width: Görüntü genişliği
            height: Görüntü yüksekliği
            fps: Hedef FPS
            auto_exposure: Otomatik pozlama
            exposure: Manuel pozlama değeri (0-1); 0-1 arası değere sahip olabilir, 0 en kısa pozlama süresi, 1 en uzun pozlama süresi anlamına gelir. Pozlama süresi kameranın ışığa ne kadar süreyle maruz kalacağını belirler. Düşük pozlama süresi (0'a yakın) hızlı hareketleri yakalamak için iyidir, ancak düşük ışık koşullarında görüntü kalitesini düşürebilir. Yüksek pozlama süresi (1'e yakın) düşük ışık koşullarında daha iyi görüntü kalitesi sağlar, ancak hızlı hareketlerde bulanıklığa neden olabilir. Otomatik pozlama devre dışı bırakıldığında, bu değeri manuel olarak ayarlayarak istediğiniz pozlama süresini belirleyebilirsiniz.
            auto_white_balance: Otomatik white balance
            white_balance: Manuel white balance (Kelvin)
            buffer_size: Frame buffer boyutu
            reconnect_enabled: Yeniden bağlanma etkin mi
            reconnect_attempts: Yeniden bağlanma denemesi
            reconnect_delay: Denemeler arası bekleme (saniye)
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        self.auto_white_balance = auto_white_balance
        self.white_balance = white_balance
        self.buffer_size = buffer_size
        
        self.reconnect_enabled = reconnect_enabled
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
    
    def open(self) -> bool:
        """Kamerayı aç ve ayarla"""
        try:
            self.cap = cv2.VideoCapture(self.device_index)
            
            if not self.cap.isOpened():
                logger.error(f"Kamera açılamadı: device {self.device_index}")
                return False
            
            # Çözünürlük ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Buffer boyutu (düşük latency için)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Exposure ayarı
            if not self.auto_exposure and self.exposure is not None:
                # Otomatik pozlamayı kapat
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manuel mod
                # Windows'ta exposure değeri farklı ölçekte olabilir
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
                logger.info(f"Manuel exposure ayarlandı: {self.exposure}")
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Otomatik mod
                logger.info("Otomatik exposure etkin")
            
            # White balance
            if not self.auto_white_balance and self.white_balance is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.white_balance)
                logger.info(f"Manuel white balance: {self.white_balance}K")
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            
            # Gerçek çözünürlüğü kontrol et
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            if actual_width != self.width or actual_height != self.height:
                logger.warning(
                    f"Çözünürlük ayarlanamadı. "
                    f"İstenen: {self.width}x{self.height}, "
                    f"Gerçek: {actual_width}x{actual_height}"
                )
            
            logger.info(
                f"Kamera açıldı: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )
            
            self._is_opened = True
            return True
            
        except Exception as e:
            logger.error(f"Kamera açılırken hata: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Frame oku, gerekirse yeniden bağlan"""
        if not self._is_opened or self.cap is None:
            logger.warning("Kamera açık değil")
            return False, None
        
        ret, frame = self.cap.read()
        
        # Başarısızlık durumunda yeniden bağlan
        if not ret and self.reconnect_enabled:
            logger.warning("Frame okunamadı, yeniden bağlanılıyor...")
            if self._reconnect():
                ret, frame = self.cap.read()
        
        return ret, frame
    
    def _reconnect(self) -> bool:
        """Kamerayı yeniden bağla"""
        for attempt in range(self.reconnect_attempts):
            logger.info(f"Yeniden bağlanma denemesi {attempt + 1}/{self.reconnect_attempts}")
            
            self.release()
            time.sleep(self.reconnect_delay)
            
            if self.open():
                logger.info("Yeniden bağlanma başarılı")
                return True
        
        logger.error("Yeniden bağlanma başarısız")
        return False
    
    def release(self) -> None:
        """Kamerayı serbest bırak"""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            logger.info("Kamera kapatıldı")
    
    def is_opened(self) -> bool:
        """Kamera açık mı"""
        return self._is_opened and self.cap is not None and self.cap.isOpened()
    
    def get_resolution(self) -> Tuple[int, int]:
        """Kamera çözünürlüğü"""
        if not self.is_opened() or self.cap is None:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def set_property(self, prop: int, value: float) -> bool:
        """Kamera özelliği ayarla"""
        if not self.is_opened() or self.cap is None:
            return False
        
        return self.cap.set(prop, value)
    
    def get_property(self, prop: int) -> float:
        """Kamera özelliği oku"""
        if not self.is_opened() or self.cap is None:
            return -1.0
        
        return self.cap.get(prop)
    
    @classmethod
    def from_config(cls, config: "CameraSettings") -> "OpenCVCamera":
        """Konfigürasyondan kamera oluştur"""
        return cls(
            device_index=config.device_index,
            width=config.width,
            height=config.height,
            fps=config.fps,
            auto_exposure=config.auto_exposure,
            exposure=config.exposure,
            auto_white_balance=config.auto_white_balance,
            white_balance=config.white_balance,
            buffer_size=config.buffer_size,
            reconnect_enabled=config.reconnect_enabled,
            reconnect_attempts=config.reconnect_attempts,
            reconnect_delay=config.reconnect_delay_sec,
        )
