"""RF-DETR model eğitim scripti"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

# rfdetr deprecation uyarılarını bastır (kütüphane içi)
warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
warnings.filterwarnings("ignore", category=UserWarning, module="rfdetr")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import build_configs
from src.utils.logging import setup_logger


def train_rf_detr(
    coco_data_dir: Path,
    output_dir: Path,
    model_size: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    resolution: int,
    device: str,
    final_model_path: Optional[Path] = None,
    patience: int = 20,
):
    """RF-DETR modelini eğit — COCO formatında veri seti gerektirir."""
    logger = logging.getLogger(__name__)

    try:
        from rfdetr import RFDETRBase, RFDETRLarge
    except ImportError:
        logger.error("rfdetr kütüphanesi bulunamadı. Yüklemek için: pip install rfdetr")
        return None

    if not coco_data_dir.exists():
        logger.error(f"COCO veri seti dizini bulunamadı: {coco_data_dir}")
        logger.error("RF-DETR, COCO formatında train/ ve valid/ alt dizinleri içeren veri seti gerektirir.")
        return None

    model_cls = RFDETRLarge if model_size.lower() == "large" else RFDETRBase
    logger.info(f"RF-DETR{model_size.capitalize()} başlatılıyor (resolution={resolution}, device={device}, num_classes=1)")
    model = model_cls(device=device, resolution=resolution, num_classes=1)

    logger.info("Eğitim başlıyor...")
    logger.info(f"  Data: {coco_data_dir}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch: {batch_size}  (grad_accum_steps={grad_accum_steps})")
    logger.info(f"  LR: {lr}")
    logger.info(f"  Device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        dataset_dir=str(coco_data_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=str(output_dir),
        patience=patience,
    )

    logger.info("RF-DETR eğitimi tamamlandı")

    # En iyi checkpoint'i hedef yola kopyala
    best_ckpt = output_dir / "checkpoint_best_total.pth"
    if not best_ckpt.exists():
        # Alternatif isim
        candidates = list(output_dir.glob("*.pth"))
        if candidates:
            best_ckpt = max(candidates, key=lambda p: p.stat().st_mtime)

    if best_ckpt.exists():
        logger.info(f"En iyi model: {best_ckpt}")
        if final_model_path is not None:
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(best_ckpt, final_model_path)
            logger.info(f"Model kopyalandı: {final_model_path}")

    return model


def main():
    configs = build_configs()
    logger = setup_logger("train", level=configs.base.logging.level)
    logger.info("=" * 60)
    logger.info("TEKNOFEST Robolig AI - RF-DETR Dedektör Eğitimi")
    logger.info("=" * 60)

    device = configs.base.system.device

    try:
        rf = configs.data.training.rf_detr
        coco_data_dir = ROOT / rf.coco_data_dir
        output_dir = ROOT / rf.save_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results = train_rf_detr(
            coco_data_dir=coco_data_dir,
            output_dir=output_dir,
            model_size=configs.detection.rf_detr.model_size,
            epochs=rf.epochs,
            batch_size=rf.batch_size,
            grad_accum_steps=rf.grad_accum_steps,
            lr=rf.lr,
            resolution=rf.resolution,
            device=device,
            final_model_path=ROOT / configs.detection.rf_detr.model_path,
            patience=rf.patience,
        )

        if results is None:
            return 1

        logger.info("Eğitim başarıyla tamamlandı")
        return 0

    except Exception as e:
        logger.error(f"Eğitim hatası: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
