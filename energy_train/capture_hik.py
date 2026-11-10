"""Hikrobot MVS color source. The SDK is loaded only when the camera starts."""

import ctypes
import importlib
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .capture import CaptureConfig

# GenICam enum values used by Hikrobot cameras.
AUTO_OFF, AUTO_CONTINUOUS = 0, 2  # ExposureAuto / GainAuto
WHITE_BALANCE_OFF, WHITE_BALANCE_CONTINUOUS = 0, 1  # BalanceWhiteAuto
FRAME_TIMEOUT_MS = 5000
MAX_INCOMPLETE_FRAMES = 10


@dataclass
class HikCaptureConfig(CaptureConfig):
    width: int = 1440
    height: int = 1080
    exposure_auto: bool = True
    exposure_time: float = 5000.0  # microseconds, used when exposure_auto is false
    gain_auto: bool = True
    gain: float = 0.0  # dB, used when gain_auto is false
    balance_white_auto: bool = True
    sdk_root: str = "/opt/MVS"

    def validate(self):
        super().validate()
        for name in ("exposure_auto", "gain_auto", "balance_white_auto"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须为 true/false")
        for name, positive in (("exposure_time", True), ("gain", False)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) \
                    or value < 0 or (positive and value == 0):
                raise ValueError(f"{name} 必须为{'正' if positive else '非负'}有限数值")
        if not isinstance(self.sdk_root, str) or not self.sdk_root:
            raise ValueError("sdk_root 必须为 MVS 安装目录")


def load_sdk(sdk_root):
    """Import MvCameraControl_class from the MVS install (it is not a pip package)."""
    root = Path(sdk_root).expanduser()
    arch = {"x86_64": "64", "amd64": "64", "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine().lower())
    candidates = [root / "Samples" / arch / "Python" / "MvImport"] if arch else []
    candidates += sorted(root.glob("Samples/*/Python/MvImport"))
    folder = next((path for path in candidates if (path / "MvCameraControl_class.py").is_file()), None)
    if folder is None:
        raise RuntimeError(f"未找到海康 MVS SDK 的 Python 接口（{root}/Samples/*/Python/MvImport）；"
                           "请安装 MVS 或在 capture.yaml 的 hik.sdk_root 指定安装目录")
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(root / "lib"))
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        return importlib.import_module("MvCameraControl_class")
    except OSError as exc:
        raise RuntimeError(f"无法加载 libMvCameraControl.so，检查 MVCAM_COMMON_RUNENV 或 MVS 安装: {exc}") from exc


def _text(array):
    return bytes(array).split(b"\0", 1)[0].decode(errors="replace")


def _check(ret, action):
    if ret != 0:
        raise RuntimeError(f"海康 SDK {action}失败 (错误码 0x{ret & 0xFFFFFFFF:08X})")


class HikColorSource:
    def __init__(self, config):
        self.config = config
        self.mv = self.cam = None
        self._initialized = self._handle = self._open = self._grabbing = False

    def _enumerate(self):
        mv = self.mv
        devices = mv.MV_CC_DEVICE_INFO_LIST()
        _check(mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE, devices), "枚举设备")
        found = []
        for index in range(devices.nDeviceNum):
            info = ctypes.cast(devices.pDeviceInfo[index], ctypes.POINTER(mv.MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType == mv.MV_USB_DEVICE:
                special, interface = info.SpecialInfo.stUsb3VInfo, "USB3"
            elif info.nTLayerType == mv.MV_GIGE_DEVICE:
                special, interface = info.SpecialInfo.stGigEInfo, "GigE"
            else:
                continue
            found.append((_text(special.chSerialNumber), _text(special.chModelName), interface, info))
        return found

    def _int(self, name):
        value = self.mv.MVCC_INTVALUE_EX()
        _check(self.cam.MV_CC_GetIntValueEx(name, value), f"读取 {name} ")
        return value

    def _set(self, kind, name, value):
        setter = {"int": self.cam.MV_CC_SetIntValueEx, "enum": self.cam.MV_CC_SetEnumValue,
                  "float": self.cam.MV_CC_SetFloatValue, "bool": self.cam.MV_CC_SetBoolValue}[kind]
        _check(setter(name, value), f"设置 {name}={value} ")

    def start(self):
        cfg = self.config
        self.mv = mv = load_sdk(cfg.sdk_root)
        if hasattr(mv.MvCamera, "MV_CC_Initialize"):
            _check(mv.MvCamera.MV_CC_Initialize(), "初始化")
            self._initialized = True
        found = self._enumerate()
        if cfg.serial:
            matches = [item for item in found if item[0] == cfg.serial]
        else:
            matches = found
        if len(matches) != 1:
            listed = ", ".join(f"{sn}({model})" for sn, model, _, _ in found) or "无"
            reason = f"未找到 SN={cfg.serial}" if cfg.serial else "未指定 --serial"
            raise RuntimeError(f"{reason}；已发现海康相机: {listed}。检查 USB/网线，或关闭 MVS 客户端后重试")
        serial, model, interface, info = matches[0]

        self.cam = cam = mv.MvCamera()
        _check(cam.MV_CC_CreateHandle(info), "创建句柄")
        self._handle = True
        _check(cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0), "打开相机（是否被 MVS 客户端或其他进程占用）")
        self._open = True
        if interface == "GigE":
            packet = cam.MV_CC_GetOptimalPacketSize()
            if packet > 0:
                self._set("int", "GevSCPSPacketSize", packet)

        self._set("enum", "TriggerMode", mv.MV_TRIGGER_MODE_OFF)
        width_max, height_max = self._int("WidthMax").nCurValue, self._int("HeightMax").nCurValue
        if cfg.width > width_max or cfg.height > height_max:
            raise RuntimeError(f"分辨率 {cfg.width}x{cfg.height} 超过传感器 {width_max}x{height_max}")
        # Shrink offsets first so the new width/height are always within range; smaller sizes are centered.
        self._set("int", "OffsetX", 0)
        self._set("int", "OffsetY", 0)
        self._set("int", "Width", cfg.width)
        self._set("int", "Height", cfg.height)
        offsets = {}
        for name, free in (("OffsetX", width_max - cfg.width), ("OffsetY", height_max - cfg.height)):
            step = max(self._int(name).nInc, 1)
            offsets[name] = free // 2 // step * step
            if offsets[name]:
                self._set("int", name, offsets[name])
        self._set("enum", "PixelFormat", mv.PixelType_Gvsp_BGR8_Packed)
        self._set("bool", "AcquisitionFrameRateEnable", True)
        self._set("float", "AcquisitionFrameRate", float(cfg.fps))
        self._set("enum", "ExposureAuto", AUTO_CONTINUOUS if cfg.exposure_auto else AUTO_OFF)
        if not cfg.exposure_auto:
            self._set("float", "ExposureTime", float(cfg.exposure_time))
        self._set("enum", "GainAuto", AUTO_CONTINUOUS if cfg.gain_auto else AUTO_OFF)
        if not cfg.gain_auto:
            self._set("float", "Gain", float(cfg.gain))
        self._set("enum", "BalanceWhiteAuto",
                  WHITE_BALANCE_CONTINUOUS if cfg.balance_white_auto else WHITE_BALANCE_OFF)
        _check(cam.MV_CC_SetGrabStrategy(mv.MV_GrabStrategy_LatestImagesOnly), "设置取流策略")
        _check(cam.MV_CC_StartGrabbing(), "开始取流")
        self._grabbing = True
        return {"width": cfg.width, "height": cfg.height, "fps": cfg.fps, "serial": serial,
                "name": f"Hikrobot {model}", "interface": interface, "pixel_format": "BGR8",
                "sensor": {"width_max": width_max, "height_max": height_max,
                           "offset_x": offsets["OffsetX"], "offset_y": offsets["OffsetY"]},
                "intrinsics": None}

    def read(self):
        mv, cfg = self.mv, self.config
        expected = cfg.width * cfg.height * 3
        for _ in range(MAX_INCOMPLETE_FRAMES):
            frame = mv.MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
            _check(self.cam.MV_CC_GetImageBuffer(frame, FRAME_TIMEOUT_MS), "取帧（超时或相机断开）")
            try:
                info = frame.stFrameInfo
                if info.nLostPacket:
                    continue  # incomplete transfer; wait for the next frame
                if info.enPixelType != mv.PixelType_Gvsp_BGR8_Packed or (info.nWidth, info.nHeight) != (cfg.width, cfg.height):
                    raise RuntimeError(f"收到意外的帧格式 0x{info.enPixelType:X} {info.nWidth}x{info.nHeight}")
                if max(info.nFrameLen, info.nFrameLenEx) < expected:
                    raise RuntimeError("帧数据长度不足")
                data = np.ctypeslib.as_array(frame.pBufAddr, shape=(expected,))
                return data.reshape(cfg.height, cfg.width, 3).copy(), float(info.nHostTimeStamp)
            finally:
                self.cam.MV_CC_FreeImageBuffer(frame)
        raise RuntimeError(f"连续 {MAX_INCOMPLETE_FRAMES} 帧丢包，检查 USB3 线缆/接口带宽")

    def stop(self):
        cam = self.cam
        if self._grabbing:
            cam.MV_CC_StopGrabbing()
            self._grabbing = False
        if self._open:
            cam.MV_CC_CloseDevice()
            self._open = False
        if self._handle:
            cam.MV_CC_DestroyHandle()
            self._handle = False
        if self._initialized:
            self.mv.MvCamera.MV_CC_Finalize()
            self._initialized = False
