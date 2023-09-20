import yaml
from typing import ClassVar
from dataclasses import dataclass

with open("config.yaml", "r") as __config_yaml:
    _config = yaml.load(__config_yaml, Loader=yaml.FullLoader)


@dataclass
class ReconstructionsConfig:
    inside_camera_parameters_path: str = _config["reconstruction"]["inside_camera_parameters_path"]


@dataclass
class MonoConfig:
    camera_ids: ClassVar[list[int]] = _config["mono_calibration"]["camera_ids"]
    mode: str = _config["mono_calibration"]["mode"]
    width: int = _config["mono_calibration"]["width"]
    height: int = _config["mono_calibration"]["height"]
    img_count: int = _config["mono_calibration"]["img_count"]
    time_out: int = _config["mono_calibration"]["time_out"]
    path: str = _config["mono_calibration"]["path"]
    show_gui: bool = _config["mono_calibration"]["show_gui"]
    rows: int = _config["mono_calibration"]["rows"]
    columns: int = _config["mono_calibration"]["columns"]
    type_calib: str = _config["mono_calibration"]["type_calib"]
    save: bool = _config["mono_calibration"]["save"]


@dataclass
class StereoConfig:
    camera_ids: ClassVar[list[int]] = _config["stereo_calibration"]["camera_ids"]
    mode: str = _config["stereo_calibration"]["mode"]
    width: int = _config["stereo_calibration"]["width"]
    height: int = _config["stereo_calibration"]["height"]
    img_count: int = _config["stereo_calibration"]["img_count"]
    time_out: int = _config["stereo_calibration"]["time_out"]
    path: str = _config["stereo_calibration"]["path"]
    show_gui: bool = _config["stereo_calibration"]["show_gui"]
    rows: int = _config["stereo_calibration"]["rows"]
    columns: int = _config["stereo_calibration"]["columns"]
    type_calib: str = _config["stereo_calibration"]["type_calib"]
    save: bool = _config["stereo_calibration"]["save"]


@dataclass
class DetectionConfig:
    detection_confidence: float = _config["detection"]["min_detection_confidence"]
    tracking_confidence: float = _config["detection"]["min_tracking_confidence"]
    num_hands: int = _config["detection"]["max_num_hands"]


@dataclass
class CameraConfig:
    ids: ClassVar[list[int]] = _config["camera"]["ids"]
    mode: str = _config["camera"]["mode"]
    width: int = _config["camera"]["width"]
    height: int = _config["camera"]["height"]


@dataclass
class FingersID:
    wrist: ClassVar[list[int]] = _config["fingers"]["wrist"]
    thumb: ClassVar[list[int]] = _config["fingers"]["thumb"]
    index: ClassVar[list[int]] = _config["fingers"]["index"]
    middle: ClassVar[list[int]] = _config["fingers"]["middle"]
    ring: ClassVar[list[int]] = _config["fingers"]["ring"]
    pinky: ClassVar[list[int]] = _config["fingers"]["pinky"]