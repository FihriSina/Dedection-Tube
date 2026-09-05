"""YAML tabanlı konfigürasyon yönetimi"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """YAML konfigürasyon dosyalarını yükler ve birleştirir"""
    
    def __init__(self, config_dir: Path):
        """
        Args:
            config_dir: Konfigürasyon dosyalarının bulunduğu dizin
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config dizini bulunamadı: {self.config_dir}")
        
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Belirtilen konfigürasyon dosyasını yükler
        
        Args:
            config_name: Config dosyası adı (.yaml uzantısı opsiyonel)
            use_cache: Cache kullanılsın mı
        
        Returns:
            Konfigürasyon dictionary
        """
        if not config_name.endswith(".yaml"):
            config_name = f"{config_name}.yaml"
        
        if use_cache and config_name in self._cache:
            return self._cache[config_name]
        
        config_path = self.config_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            self._cache[config_name] = config
            logger.info(f"Konfigürasyon yüklendi: {config_name}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parse hatası: {config_path} - {e}")
            raise
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Config dizinindeki tüm YAML dosyalarını yükler
        
        Returns:
            Tüm konfigürasyonları içeren dictionary
        """
        all_configs = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            all_configs[config_name] = self.load(config_name)
        
        return all_configs
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Birden fazla konfigürasyonu birleştirir (sonraki öncelikli)
        
        Args:
            *config_names: Birleştirilecek config dosya adları
        
        Returns:
            Birleştirilmiş konfigürasyon
        """
        merged = {}
        
        for config_name in config_names:
            config = self.load(config_name)
            merged = self._deep_merge(merged, config)
        
        return merged
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """İki dictionary'yi derinlemesine birleştirir"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


def get_project_root() -> Path:
    """Proje kök dizinini bulur"""
    current = Path(__file__).resolve()
    
    # src/utils/config.py -> proje kökü
    return current.parent.parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """Hızlı config yükleme yardımcı fonksiyonu"""
    config_dir = get_project_root() / "config"
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)


class ConfigBuilder:
    """YAML dosyalarını yükleyip tam tipli AllConfigs nesnesi oluşturur."""

    def __init__(self, config_dir: Path):
        self._loader = ConfigLoader(config_dir)

    def build(self) -> "AllConfigs":
        from src.utils.config_models import (
            AllConfigs, BaseConfig, CameraSettings, CalibrationConfig,
            DataFileConfig, DemoConfig, DetectionConfig,
            OutputConfig, PoseConfig,
        )
        base_raw      = self._loader.load("base")
        camera_raw    = self._loader.load("camera")
        data_raw      = self._loader.load("data")
        demo_raw      = self._loader.load("demo")
        detection_raw = self._loader.load("detection")
        output_raw    = self._loader.load("output")
        pose_raw      = self._loader.load("pose")

        return AllConfigs(
            base        = BaseConfig.model_validate(base_raw),
            camera      = CameraSettings.model_validate(camera_raw.get("camera", {})),
            calibration = CalibrationConfig.model_validate(camera_raw.get("calibration", {})),
            data        = DataFileConfig.model_validate(data_raw),
            demo        = DemoConfig.model_validate(demo_raw.get("demo", {})),
            detection   = DetectionConfig.model_validate(detection_raw.get("detection", {})),
            output      = OutputConfig.model_validate(output_raw.get("output", {})),
            pose        = PoseConfig.model_validate(pose_raw.get("pose", {})),
        )


def build_configs(config_dir: Optional[Path] = None) -> "AllConfigs":
    """Tüm YAML config dosyalarından tam tipli AllConfigs nesnesi döndürür.

    Kullanım:
        configs = build_configs()
        configs.data.training.rf_detr.epochs            # 100
        configs.camera.device_index                     # 0
        configs.detection.rf_detr.confidence_threshold  # 0.5
        configs.pose.pnp.min_depth_m                    # 0.2
    """
    if config_dir is None:
        config_dir = get_project_root() / "config"
    return ConfigBuilder(config_dir).build()


# AllConfigs tipini döngüsel import olmadan tip denetimi için sağla
def __getattr__(name: str):
    if name == "AllConfigs":
        from src.utils.config_models import AllConfigs
        return AllConfigs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
