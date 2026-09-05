"""Gerçek zamanlı kamera demo scripti."""

import argparse # argparse modülü, komut satırı argümanlarını kolayca tanımlamak ve işlemek için kullanılır. Bu, kullanıcıların scripti çalıştırırken çeşitli seçenekler ve parametreler sağlamasına olanak tanır.
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.io import load_calibration
from src.calibration.undistort import Undistorter
from src.camera.opencv_camera import OpenCVCamera
from src.detection.fusion import FusionDetector
from src.pipeline.assembler import ResultAssembler
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.visualizer import Visualizer
from src.tracking import build_tracker_from_config
from src.utils.config import build_configs, get_project_root
from src.utils.logging import setup_logger

# TEKNOFEST 2026 Robolig AI - Gerçek Zamanlı Demo
def main():
    parser = argparse.ArgumentParser( #Parser = argparse.ArgumentParser() ifadesi, komut satırı argümanlarını tanımlamak ve işlemek için kullanılan bir argparse.ArgumentParser nesnesi oluşturur. Bu nesne, scriptin kullanıcı tarafından sağlanan argümanları tanımasına ve işlemesine olanak tanır.
        description="TEKNOFEST Robolig AI - Gerçek Zamanlı Demo"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Konfigürasyon dizini (varsayılan: proje_kök/config)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="camera_calibration.npz",
        help="Kalibrasyon dosyası",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Görselleştirmeyi kapat",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default=None,
        choices=["jsonl", "serial", "socket", "stdout"],
        help="Çıktı iletim modu",
    )

    args = parser.parse_args()

    logger = setup_logger("demo", level=args.log_level)
    logger.info("=" * 60)
    logger.info("TEKNOFEST 2026 Robolig AI - Gerçek Zamanlı Demo")
    logger.info("=" * 60)

    try:
        config_dir = Path(args.config_dir) if args.config_dir else None
        configs = build_configs(config_dir)
        logger.info("Konfigürasyon yüklendi")
    except Exception as e:
        logger.error(f"Config yüklenemedi: {e}")
        return 1

    calibration_path = Path(args.calibration)
    calibration = load_calibration(calibration_path)

    if calibration is None:
        logger.warning("Kalibrasyon yüklenemedi, pose tahmini devre dışı")
        undistorter = None
    else:
        logger.info(f"Kalibrasyon yüklendi: {calibration_path}")
        undistorter = Undistorter(calibration)

    logger.info("Modüller yükleniyor...")

    camera = OpenCVCamera.from_config(configs.camera)
    if not camera.open():
        logger.error("Kamera açılamadı")
        return 1

    detector = FusionDetector.from_config(configs.detection)
    logger.info(f"Dedektör: {detector.get_name()}")
    try:
        neural_name = detector.neural.get_name()
        logger.info(f"Neural dedektör: {neural_name}")
    except Exception:
        logger.debug("Neural dedektör bilgisi alınamadı")
    try:
        classical_name = detector.classical.get_name()
        logger.info(f"Klasik dedektör: {classical_name}")
    except Exception:
        logger.debug("Klasik dedektör bilgisi alınamadı")

    tracker = build_tracker_from_config(configs.detection.tracking)
    logger.info(
        "Tracking: %s",
        configs.detection.tracking.tracker_type if configs.detection.tracking.enabled else "disabled",
    )

    assembler = ResultAssembler(
        cube_size_m=configs.base.cube.size_m,
        use_pnp=True,
        pnp_fallback_to_geometric=True,
    )

    visualizer = Visualizer.from_config(configs.demo) if not args.no_visualize else None
    runtime = PipelineRuntime(configs=configs, output_mode=args.output_mode)

    logger.info("Demo başladı. Çıkmak için 'q' tuşuna basın.")

    frame_id = 0
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = camera.read()
                if not ret or frame is None:
                    logger.warning("Frame okunamadı")
                    break

                frame_id += 1
                start_time = time.time()

                if undistorter is not None:
                    frame = undistorter.undistort(frame)

                detections = detector.detect(frame)
                detections = tracker.update(image=frame, detections=detections)
                results = assembler.assemble(
                    image=frame,
                    detections=detections,
                    calibration=calibration,
                    frame_id=frame_id,
                    timestamp=time.time(),
                )

                processing_time = time.time() - start_time
                fps, _ = runtime.emit(results)

                if visualizer is not None:
                    display = visualizer.draw_results(frame, results)
                    cv2.putText(
                        display,
                        f"FPS: {fps:.1f}",
                        (display.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("TEKNOFEST Robolig AI", display)

                if frame_id % 30 == 0:
                    logger.info(
                        f"Frame {frame_id}: {len(results.cubes)} küp | "
                        f"FPS: {fps:.1f} | Latency: {processing_time * 1000:.1f}ms"
                    )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Kullanıcı çıkış yaptı")
                break
            elif key == ord(" "):
                paused = not paused
                logger.info(f"{'Durduruldu' if paused else 'Devam'}")
            elif key == ord("s"):
                save_path = f"saved_frame_{frame_id}.jpg"
                cv2.imwrite(save_path, frame)
                logger.info(f"Frame kaydedildi: {save_path}")

    except KeyboardInterrupt:
        logger.info("Klavye ile durduruldu")
    except Exception as e:
        logger.error(f"Demo hatası: {e}", exc_info=True)
        return 1
    finally:
        camera.release()
        runtime.close()
        cv2.destroyAllWindows()
        logger.info("Demo sonlandı")

    return 0


if __name__ == "__main__":
    sys.exit(main())