"""Raw BGR collection. Camera, preview and storage have separate boundaries."""

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import atomic_json, file_lock, read_json, utc_now, valid_name


@dataclass
class CaptureConfig:
    session: str = ""
    group_id: str = ""
    output_root: Path = Path("raw")
    width: int = 640
    height: int = 480
    fps: int = 30
    interval: float = 0.5
    warmup_frames: int = 30
    max_images: int = 0
    duration: float = 0
    preview: bool = True
    auto: bool = True
    blur_threshold: float = 50
    reject_blur: bool = False
    duplicate_threshold: float = 2
    serial: str | None = None
    note: str = ""

    def validate(self):
        valid_name(self.session)
        valid_name(self.group_id)
        for name in ("width", "height", "fps", "warmup_frames", "max_images"):
            value = getattr(self, name)
            minimum = 1 if name in {"width", "height", "fps"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} 必须为 >= {minimum} 的整数")
        for name in ("interval", "duration", "blur_threshold", "duplicate_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为非负有限数值")
        if self.interval == 0:
            raise ValueError("interval 必须大于 0")
        for name in ("preview", "auto", "reject_blur"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须为 true/false")
        if not self.preview and not self.auto:
            raise ValueError("无窗口模式必须启用自动采集")


def assess_frame(frame, previous=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = None
    if previous is not None:
        difference = float(cv2.absdiff(frame, previous).mean())
    return {"blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "difference": difference}


def rejection_reason(quality, config, manual=False):
    if manual:
        return None
    if config.reject_blur and quality["blur_score"] < config.blur_threshold:
        return "blur"
    difference = quality["difference"]
    if difference is not None and difference < config.duplicate_threshold:
        return "duplicate"
    return None


class CaptureSession:
    """Single writer per session. PNG creation is exclusive, never overwriting."""

    def __init__(self, root, session, group_id, camera, settings):
        self.session, self.group_id = valid_name(session), valid_name(group_id)
        self.path = Path(root) / session
        self.camera, self.settings = camera, settings
        self._lock = None

    def __enter__(self):
        self._lock = file_lock(self.path / ".capture.lock")
        self._lock.__enter__()
        try:
            meta_path = self.path / "meta.json"
            if meta_path.exists():
                meta = read_json(meta_path)
                if meta["session"] != self.session or meta["group_id"] != self.group_id:
                    raise ValueError("续拍 session/group_id 与原始记录不一致")
                for key in ("width", "height", "serial"):
                    if meta["camera"].get(key) != self.camera.get(key):
                        raise ValueError(f"续拍 camera.{key} 改变，请建立新 session")
            else:
                if any(self.path.glob("*.png")) or (self.path / "frames.jsonl").exists():
                    raise ValueError("已有图片/日志但缺少 meta.json，请先恢复元数据或使用新 session")
                meta = {"schema_version": 1, "session": self.session, "group_id": self.group_id,
                        "created_at": utc_now(), "camera": self.camera,
                        "color_order": "BGR", "depth_saved": False, "capture_runs": []}
            pattern = re.compile(re.escape(self.session) + r"_(\d+)\.png$")
            numbers = [int(m.group(1)) for p in self.path.glob("*.png") if (m := pattern.fullmatch(p.name))]
            self.next_number = max(numbers, default=-1) + 1
            meta["capture_runs"].append({"started_at": utc_now(), "settings": self.settings})
            atomic_json(meta_path, meta)
            return self
        except BaseException:
            self._lock.__exit__(None, None, None)
            self._lock = None
            raise

    def __exit__(self, *args):
        if self._lock:
            self._lock.__exit__(*args)
            self._lock = None

    def save(self, frame, timestamp_ms, reason, quality):
        if self._lock is None:
            raise RuntimeError("CaptureSession 必须在 with 中使用")
        if frame.shape != (self.camera["height"], self.camera["width"], 3) or frame.dtype != np.uint8:
            raise ValueError("输入必须是与相机尺寸一致的 uint8 BGR 图像")
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise OSError("PNG 编码失败")
        payload = encoded.tobytes()
        path = self.path / f"{self.session}_{self.next_number:05d}.png"
        record = {"image": path.name, "session": self.session, "group_id": self.group_id,
                  "saved_at": utc_now(), "timestamp_ms": timestamp_ms, "reason": reason,
                  "quality": quality, "image_sha256": hashlib.sha256(payload).hexdigest()}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        # Exclusive create protects even against an uncooperative concurrent writer.
        with path.open("xb") as stream:
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        try:
            with (self.path / "frames.jsonl").open("a+", encoding="utf-8") as log:
                offset = log.tell()
                try:
                    log.write(line)
                    log.flush()
                    os.fsync(log.fileno())
                except BaseException:
                    log.seek(offset)
                    log.truncate()
                    raise
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self.next_number += 1
        return path


class RealSenseColorSource:
    def __init__(self, config):
        self.config = config
        self.pipeline = None

    def start(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("缺少 pyrealsense2，请安装 requirements/capture.txt") from exc
        cfg = self.config
        self.pipeline = rs.pipeline()
        settings = rs.config()
        if cfg.serial:
            settings.enable_device(cfg.serial)
        settings.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
        try:
            profile = self.pipeline.start(settings)
        except RuntimeError as exc:
            raise RuntimeError("无法启动 D435 彩色流：检查 USB、权限、分辨率及是否被 ROS/viewer 占用。" + str(exc)) from exc
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        dev = profile.get_device()
        return {"width": intr.width, "height": intr.height, "fps": cfg.fps,
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
                "intrinsics": {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                               "model": str(intr.model), "coeffs": list(intr.coeffs)}}

    def read(self):
        frame = self.pipeline.wait_for_frames(timeout_ms=5000).get_color_frame()
        if not frame:
            raise RuntimeError("未收到有效彩色帧")
        return np.asanyarray(frame.get_data()).copy(), frame.get_timestamp()

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass  # start may have failed before a stream was active


def run_capture(config, source=None, clock=time.monotonic):
    config.validate()
    if config.preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("没有图形显示服务，请使用 --no-preview --max-images N 或 --duration 秒数")
    source = source or RealSenseColorSource(config)
    stats = {"saved": 0, "blur": 0, "duplicate": 0, "interrupted": False}
    previous = None
    auto = config.auto
    window = "Energy unit capture | Space save | A auto/pause | Q quit"
    window_open = False
    settings = asdict(config)
    settings["output_root"] = str(config.output_root)
    try:
        camera = source.start()
        with CaptureSession(config.output_root, config.session, config.group_id, camera, settings) as session:
            start = clock()
            for _ in range(config.warmup_frames):
                source.read()
            if config.preview:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                window_open = True
            last_attempt = None
            while not config.max_images or stats["saved"] < config.max_images:
                frame, timestamp = source.read()
                now = clock()
                if config.duration and now - start >= config.duration:
                    break
                quality = assess_frame(frame, previous)
                manual = False
                if config.preview:
                    preview = frame.copy()
                    lines = [f"{config.session} | {'AUTO' if auto else 'PAUSED'} | saved={stats['saved']}",
                             f"blur={quality['blur_score']:.1f} | skipped blur={stats['blur']} dup={stats['duplicate']}"]
                    for i, line in enumerate(lines):
                        cv2.putText(preview, line, (8, 24 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                    cv2.imshow(window, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("a"):
                        auto = not auto
                    manual = key == ord(" ")
                due = auto and (last_attempt is None or now - last_attempt >= config.interval)
                if manual or due:
                    last_attempt = now
                    rejection = rejection_reason(quality, config, manual)
                    if rejection:
                        stats[rejection] += 1
                    else:
                        path = session.save(frame, timestamp, "manual" if manual else "auto", quality)
                        previous = frame.copy()
                        stats["saved"] += 1
                        print(f"saved {path}", flush=True)
    except KeyboardInterrupt:
        stats["interrupted"] = True
    finally:
        source.stop()
        if window_open:
            cv2.destroyWindow(window)
    return stats
