"""Kamera arayüzü - abstract base class"""
# interface.py, kamera erişimi için bir arayüz tanımlar. CameraInterface sınıfı, farklı kamera türleri (örneğin, OpenCV tabanlı, gerçek donanım kameraları) için ortak bir arayüz sağlar. Bu sınıf, kamerayı açma, frame okuma, serbest bırakma ve özellik ayarlama gibi temel işlemleri tanımlar. Böylece, pipeline içinde farklı kamera implementasyonları sorunsuz bir şekilde kullanılabilir ve gerektiğinde kolayca değiştirilebilir.
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class CameraInterface(ABC):
    """Kamera erişimi için temel arayüz"""
    
    @abstractmethod
    def open(self) -> bool:
        """
        Kamerayı aç
        
        Returns:
            Başarılı ise True
        """
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Frame oku
        
        Returns:
            (başarı durumu, frame) tuple
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Kamerayı serbest bırak"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Kamera açık mı"""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Kamera çözünürlüğü
        
        Returns:
            (genişlik, yükseklik)
        """
        pass
    
    @abstractmethod
    def set_property(self, prop: int, value: float) -> bool:
        """
        Kamera özelliği ayarla
        
        Args:
            prop: OpenCV property kodu
            value: Değer
        
        Returns:
            Başarılı ise True
        """
        pass
