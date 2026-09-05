"""Dedektör temel sınıfı"""
#base.py , tüm dedektörlerin uyması gereken ortak bir arayüz tanımlar. BaseDetector sınıfı, detect ve get_name yöntemlerini soyut olarak tanımlar. detect yöntemi, verilen bir görüntü üzerinde nesne algılama işlemini gerçekleştirmek için kullanılırken, get_name yöntemi dedektörün adını döndürür. Bu yapı, farklı dedektör türlerinin aynı arayüzü kullanarak pipeline içinde sorunsuz bir şekilde çalışmasını sağlar.
from abc import ABC, abstractmethod
import numpy as np

from src.detection.models import DetectorOutput


class BaseDetector(ABC):
    """Algılama için temel sınıf"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectorOutput:
        """
        Görüntüde nesne algıla
        
        Args:
            image: BGR formatında görüntü
        
        Returns:
            Algılama sonuçları
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Dedektör adı"""
        pass
