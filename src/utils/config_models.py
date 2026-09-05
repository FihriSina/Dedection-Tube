"""Tipli config modelleri - Pydantic BaseModel (YAML → model otomatik dönüşüm)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# base.yaml
# ---------------------------------------------------------------------------

class CubeConfig(BaseModel):
    size_cm: float = 9.0
    size_m: float = 0.09
    model_points: List[List[float]] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    use_colors: bool = True
    log_dir: str = "logs"


class PerformanceConfig(BaseModel):
    target_fps: int = 20
    max_latency_ms: int = 50


class SystemConfig(BaseModel):
    random_seed: int = 42
    device: str = "cuda"


class BaseConfig(BaseModel):
    cube: CubeConfig = Field(default_factory=CubeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)


# ---------------------------------------------------------------------------
# camera.yaml
# ---------------------------------------------------------------------------

class CameraSettings(BaseModel):
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    auto_exposure: bool = False
    exposure: float = 0.25
    auto_white_balance: bool = False
    white_balance: int = 4600
    calibration_file: str = "camera_calibration.npz"
    use_undistort: bool = True
    reconnect_enabled: bool = True
    reconnect_attempts: int = 3
    reconnect_delay_sec: float = 2.0
    buffer_size: int = 1
    synthetic_focal_length_px: float = 800.0


class CalibrationConfig(BaseModel):
    checkerboard_size: List[int] = Field(default_factory=lambda: [9, 6])
    square_size_mm: int = 25
    min_images: int = 15
    max_images: int = 30
    max_reprojection_error: float = 0.5


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

class DataPathsConfig(BaseModel):
    dataset_root: str = "data/processed/coco"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    unified_dir: str = "data/processed/unified"
    stats_json: str = "data/processed/unified/stats.json"
    stats_markdown: str = "data/processed/unified/dataset_report.md"
    synthetic_dir: str = "data/synthetic"
    real_dir: str = "data/real"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    external_sources: Dict[str, Any] = Field(default_factory=dict)


class RfDetrTrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    resolution: int = 560
    patience: int = 20
    save_dir: str = "runs/train_rf_detr"
    coco_data_dir: str = "data/processed/coco"


class TrainingConfig(BaseModel):
    rf_detr: RfDetrTrainingConfig = Field(default_factory=RfDetrTrainingConfig)


class DataFileConfig(BaseModel):
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
    classes: Dict[str, Any] = Field(default_factory=dict)
    synthetic: Dict[str, Any] = Field(default_factory=dict)
    augmentation: Dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# demo.yaml
# ---------------------------------------------------------------------------

class VisualizationColorsConfig(BaseModel):
    cube: List[int] = Field(default_factory=lambda: [0, 255, 0])
    text: List[int] = Field(default_factory=lambda: [255, 255, 255])


class VisualizationConfig(BaseModel):
    enabled: bool = True
    window_name: str = "TEKNOFEST Robolig AI"
    draw_bbox: bool = True
    draw_label: bool = True
    draw_confidence: bool = True
    draw_center: bool = True
    draw_axes: bool = True
    colors: VisualizationColorsConfig = Field(default_factory=VisualizationColorsConfig)
    font_scale: float = 0.6
    font_thickness: int = 2
    bbox_alpha: float = 0.3


class DemoOutputConfig(BaseModel):
    save_video: bool = False
    video_path: str = "output/demo_output.mp4"
    video_fps: int = 30
    video_codec: str = "mp4v"
    save_json: bool = False
    json_path: str = "output/detections.jsonl"
    save_frames: bool = False
    frames_dir: str = "output/frames"
    frame_interval: int = 30


class ModulesConfig(BaseModel):
    camera: bool = True
    detection: bool = True
    pose_estimation: bool = True


class DemoPerformanceConfig(BaseModel):
    show_fps: bool = True
    fps_window_size: int = 30
    enable_profiling: bool = False
    profile_log: str = "output/profile.log"
    skip_frames: int = 0


class ControlsConfig(BaseModel):
    quit_key: str = "q"
    pause_key: str = " "
    save_frame_key: str = "s"
    toggle_overlay_key: str = "o"
    auto_stop_after_sec: int = 0


class DemoConfig(BaseModel):
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    output: DemoOutputConfig = Field(default_factory=DemoOutputConfig)
    modules: ModulesConfig = Field(default_factory=ModulesConfig)
    performance: DemoPerformanceConfig = Field(default_factory=DemoPerformanceConfig)
    controls: ControlsConfig = Field(default_factory=ControlsConfig)


# ---------------------------------------------------------------------------
# detection.yaml
# ---------------------------------------------------------------------------

class ClassicalDetectionConfig(BaseModel):
    enabled: bool = True
    blur_kernel: int = 5
    morph_kernel: int = 5
    min_area: int = 500
    max_area: int = 100000
    min_aspect_ratio: float = 0.7
    max_aspect_ratio: float = 1.3
    min_extent: float = 0.6
    approx_epsilon: float = 0.04
    min_corners: int = 4
    max_corners: int = 8
    base_confidence: float = 0.6
    canny_threshold1: int = 50
    canny_threshold2: int = 150


class RfDetrDetectionConfig(BaseModel):
    enabled: bool = True
    model_path: str = "model/rf_detr_cube_detector.pth"
    model_size: str = "base"  # base veya large
    confidence_threshold: float = 0.5
    device: str = "cuda"
    resolution: int = 560
    classes: Dict[int, str] = Field(default_factory=dict)


class FusionConfig(BaseModel):
    prefer_neural: bool = True
    fallback_if_neural_fails: bool = True
    fallback_if_low_confidence: bool = True
    fallback_confidence_threshold: float = 0.4
    use_nms: bool = True
    nms_iou_threshold: float = 0.5


class TrackingConfig(BaseModel):
    enabled: bool = True
    tracker_type: str = "bytetrack"
    tracker_backend: str = "python"  # python veya destekleyen tracker'larda cpp
    tracker_config: Optional[str] = None
    reid_weights: Optional[str] = None
    device: Optional[str] = "cuda"
    half: bool = False
    per_class: bool = True
    keep_untracked_detections: bool = True


class DetectionConfig(BaseModel):
    rf_detr: RfDetrDetectionConfig = Field(default_factory=RfDetrDetectionConfig)
    classical: ClassicalDetectionConfig = Field(default_factory=ClassicalDetectionConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


# ---------------------------------------------------------------------------
# output.yaml
# ---------------------------------------------------------------------------
 
class OutputConfig(BaseModel):
    output_mode: str = "stdout"
    serial_port: str = "COM3"
    serial_baud: int = 115200
    socket_host: str = "localhost"
    socket_port: int = 5555
    socket_protocol: str = "tcp"
    jsonl_path: str = "output/detections.jsonl"


# ---------------------------------------------------------------------------
# pose.yaml
# ---------------------------------------------------------------------------

class CornersConfig(BaseModel):
    roi_padding: int = 6
    max_corners: int = 12
    quality_level: float = 0.01
    min_distance: int = 8
    block_size: int = 3
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    harris_threshold: float = 0.01
    approx_epsilon: float = 0.04
    canny_threshold1: int = 50
    canny_threshold2: int = 150


class PnPConfig(BaseModel):
    min_depth_m: float = 0.2
    max_depth_m: float = 3.0
    high_confidence: float = 0.9
    low_confidence: float = 0.5
    orientation_confidence: float = 0.8


class GeometryConfig(BaseModel):
    default_depth_m: float = 1.0
    depth_confidence: float = 0.6
    fallback_confidence: float = 0.1


class PoseConfig(BaseModel):
    corners: CornersConfig = Field(default_factory=CornersConfig)
    pnp: PnPConfig = Field(default_factory=PnPConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)


# ---------------------------------------------------------------------------
# Üst düzey kapsayıcı
# ---------------------------------------------------------------------------

class AllConfigs(BaseModel):
    """Tüm YAML config dosyalarını kapsayan tipli nesne.

    Kullanım:
        configs = build_configs()
        configs.data.training.rf_detr.epochs            # 100
        configs.camera.device_index                     # 0
        configs.detection.rf_detr.confidence_threshold  # 0.5
        configs.pose.pnp.min_depth_m                    # 0.2
    """
    base: BaseConfig = Field(default_factory=BaseConfig)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    data: DataFileConfig = Field(default_factory=DataFileConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
