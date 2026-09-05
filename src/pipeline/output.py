"""Pipeline sonuçlarını sabit JSON formatında dışa aktarır."""

from __future__ import annotations

import json
import logging
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from src.utils.types import (
    BoundingBox,
    CubeDetectionResult,
    FrameAnalysisResult,
)

logger = logging.getLogger(__name__)

try:
    import serial  # type: ignore
except ImportError:
    serial = None


OUTPUT_SCHEMA_VERSION = "1.0"

# round_float, verilen bir değeri belirtilen ondalık basamak sayısına yuvarlar. 
# Eğer değer None ise 0.0 döner. Bu yardımcı fonksiyon, JSON çıktısında 
# sayısal değerlerin okunabilirliğini artırmak için kullanılır.
def _round_float(value: Optional[float], digits: int = 3) -> float:
    if value is None:
        return 0.0
    return round(float(value), digits)

# bbox_to_dict, verilen bir BoundingBox nesnesini sözlüğe dönüştürür. 
# Bu, JSON çıktısında bounding box koordinatlarının kolayca temsil edilmesini sağlar.
def _bbox_to_dict(bbox: BoundingBox) -> Dict[str, int]:
    return {
        "x1": int(bbox.x1),
        "y1": int(bbox.y1),
        "x2": int(bbox.x2),
        "y2": int(bbox.y2),
    }

# cube_to_dict, verilen bir CubeDetectionResult nesnesini sözlüğe dönüştürür. Bu sözlük, küpün sınıfı, güven skoru, bounding box koordinatları, poz ve orientasyon bilgileri gibi detayları içerir. JSON çıktısında her küpün bilgilerini temsil etmek için kullanılır.
def _cube_to_dict(cube_id: int, cube: CubeDetectionResult) -> Dict[str, Any]:
    output_id = cube.detection.track_id if cube.detection.track_id is not None else cube_id
    position_cm = {"x": 0.0, "y": 0.0, "z": 0.0}
    orientation_deg = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    distance_cm = 0.0

    if cube.pose is not None:
        position_cm = {
            "x": _round_float(cube.pose.position.x * 100.0),
            "y": _round_float(cube.pose.position.y * 100.0),
            "z": _round_float(cube.pose.position.z * 100.0),
        }
        orientation_deg = {
            "yaw": _round_float(cube.pose.orientation.yaw),
            "pitch": _round_float(cube.pose.orientation.pitch),
            "roll": _round_float(cube.pose.orientation.roll),
        }
        distance_cm = _round_float(cube.pose.position.z * 100.0)

    return {
        "id": int(output_id),
        "class": cube.detection.class_name or cube.detection.class_id.value,
        "confidence": _round_float(cube.detection.confidence),
        "bbox": _bbox_to_dict(cube.detection.bbox),
        "position_cm": position_cm,
        "orientation_deg": orientation_deg,
        "distance_cm": distance_cm,
    }


def frame_result_to_payload(
    result: FrameAnalysisResult,
    fps: float = 0.0,
) -> Dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "timestamp": _round_float(result.timestamp, 6),
        "frame_id": int(result.frame_id),
        "cubes": [
            _cube_to_dict(cube_id=index, cube=cube)
            for index, cube in enumerate(result.cubes)
        ],
        "fps": _round_float(fps, 3),
    }

# BaseOutputTransport, farklı iletişim kanalları için ortak bir arayüz sağlar. Tüm özel transport sınıfları bu sınıfı genişleterek send ve close yöntemlerini uygulamalıdır. Bu, OutputPublisher'ın farklı transport türleriyle çalışmasını kolaylaştırır.
class BaseOutputTransport:
    def send(self, payload: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        return None

# JsonLinesTransport, verilen bir dosya yoluna JSON Lines formatında çıktı yazmak için kullanılan bir transport sınıfıdır. Her gönderilen payload, belirtilen dosyaya yeni bir satır olarak eklenir. Bu, sonuçların kalıcı olarak saklanması için uygundur.
class JsonLinesTransport(BaseOutputTransport):
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("a", encoding="utf-8")

    def send(self, payload: Dict[str, Any]) -> bool:
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()
        return True

    def close(self) -> None:
        self._handle.close()

# StdoutTransport, JSON formatında çıktı üretmek için standart çıktı akışını kullanan bir transport sınıfıdır. Gönderilen her payload, JSON olarak formatlanarak standart çıktıya yazılır. Bu, sonuçların anlık olarak izlenmesi için uygundur.
class StdoutTransport(BaseOutputTransport):
    def send(self, payload: Dict[str, Any]) -> bool:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return True


class SerialTransport(BaseOutputTransport):
    def __init__(self, port: str, baudrate: int):
        self._serial = None
        if serial is None:
            logger.warning("pyserial yüklü değil, serial output pasif.")
            return

        try:
            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0)
        except Exception as exc:
            logger.warning("Serial bağlantısı kurulamadı (%s): %s", port, exc)

    def send(self, payload: Dict[str, Any]) -> bool:
        if self._serial is None:
            return False

        try:
            message = json.dumps(payload, ensure_ascii=False) + "\n"
            self._serial.write(message.encode("utf-8"))
            return True
        except Exception as exc:
            logger.warning("Serial gönderimi başarısız: %s", exc)
            return False

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass


class SocketTransport(BaseOutputTransport):
    def __init__(self, host: str, port: int, protocol: str = "tcp"):
        self.host = host
        self.port = int(port)
        self.protocol = protocol.lower()
        self._socket: Optional[socket.socket] = None
        self._connect()

    def _connect(self) -> None:
        sock_type = socket.SOCK_DGRAM if self.protocol == "udp" else socket.SOCK_STREAM
        try:
            self._socket = socket.socket(socket.AF_INET, sock_type)
            if self.protocol != "udp":
                self._socket.settimeout(1.0)
                self._socket.connect((self.host, self.port))
        except Exception as exc:
            logger.warning(
                "Socket bağlantısı kurulamadı (%s://%s:%s): %s",
                self.protocol,
                self.host,
                self.port,
                exc,
            )
            self.close()

    def send(self, payload: Dict[str, Any]) -> bool:
        if self._socket is None:
            return False

        message = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            if self.protocol == "udp":
                self._socket.sendto(message, (self.host, self.port))
            else:
                self._socket.sendall(message)
            return True
        except Exception as exc:
            logger.warning("Socket gönderimi başarısız: %s", exc)
            return False

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None


class OutputPublisher:
    """Frame sonuçlarını seçilen iletişim kanalına gönderir."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mode = str(self.config.get("output_mode", "stdout")).lower()
        self.transport = self._build_transport()

    @classmethod
    def from_config(cls, config: "OutputConfig", override_mode: Optional[str] = None) -> "OutputPublisher":
        config_dict = config.model_dump()
        if override_mode:
            config_dict["output_mode"] = override_mode
        return cls(config_dict)

    def _build_transport(self) -> BaseOutputTransport:
        if self.mode == "jsonl":
            return JsonLinesTransport(str(self.config["jsonl_path"]))
        if self.mode == "serial":
            return SerialTransport(
                port=str(self.config["serial_port"]),
                baudrate=int(self.config["serial_baud"]),
            )
        if self.mode == "socket":
            return SocketTransport(
                host=str(self.config["socket_host"]),
                port=int(self.config["socket_port"]),
                protocol=str(self.config.get("socket_protocol", "tcp")),
            )
        return StdoutTransport()

    def publish(self, result: FrameAnalysisResult, fps: float = 0.0) -> Dict[str, Any]:
        payload = frame_result_to_payload(result=result, fps=fps)
        self.transport.send(payload)
        return payload

    def close(self) -> None:
        self.transport.close()
