"""Minimal RF-DETR dedektör testleri — graceful degradation ve çıktı yapısı."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.models import DetectorOutput
from src.detection.rf_detr_detector import RFDETRDetector


def test_missing_model_returns_empty_output(tmp_path):
    """Model dosyası yoksa detect() boş DetectorOutput döndürmeli, exception değil."""
    detector = RFDETRDetector(model_path=str(tmp_path / "no_model.pth"), device="cpu")
    assert detector.model is None
    output = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(output, DetectorOutput)
    assert output.is_empty()


def test_missing_model_output_lists_are_empty_not_none(tmp_path):
    """Tüm listeler None değil, boş liste olmalı."""
    detector = RFDETRDetector(model_path=str(tmp_path / "no_model.pth"), device="cpu")
    out = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert out.bboxes == []
    assert out.classes == []
    assert out.confidences == []
    assert out.class_names == []


def test_get_name_works_without_model():
    """get_name() model yüklü olmadan çalışmalı."""
    detector = RFDETRDetector(model_path="/nonexistent/model.pth", device="cpu")
    assert detector.get_name() == "RFDETRDetector"
