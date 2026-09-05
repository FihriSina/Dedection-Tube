"""Merkezi loglama sistemi"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import coloredlogs
    HAS_COLORED = True
except ImportError:
    HAS_COLORED = False


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Sistem için standart logger oluşturur
    
    Args:
        name: Logger adı
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        log_file: Opsiyonel dosya yolu
        use_colors: Renkli çıktı kullanılsın mı
    
    Returns:
        Yapılandırılmış logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Önceki handler'ları temizle
    logger.handlers.clear()
    
    # Format
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if use_colors and HAS_COLORED:
        coloredlogs.install(
            level=level.upper(),
            logger=logger,
            fmt=fmt,
            datefmt=date_fmt
        )
    else:
        formatter = logging.Formatter(fmt, datefmt=date_fmt)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Dosyaya her şeyi yaz
        file_formatter = logging.Formatter(fmt, datefmt=date_fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
