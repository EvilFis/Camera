import yaml
from typing import ClassVar
from dataclasses import dataclass

with open("config.yaml", "r") as __config_yaml:
    _config = yaml.load(__config_yaml, Loader=yaml.FullLoader)


@dataclass
class ReconstructionsConfig:
    inside_camera_parameters_path: str = _config["reconstruction"]["inside_camera_parameters_path"]


@dataclass
class CalibrationConfig:
    img_count: int = _config["calibration"]["img_count"]
    time_out: int = _config["calibration"]["time_out"]
    mono_calibration_path: str = _config["calibration"]["mono_calibration_path"]
    stereo_calibration_path: str = _config["calibration"]["stereo_calibration_path"]
    show_gui: bool = _config["calibration"]["show_gui"]
    rows: int = _config["calibration"]["rows"]
    columns: int = _config["calibration"]["columns"]
    type_calib: str = _config["calibration"]["type_calib"]
    save: bool = _config["calibration"]["save"]


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