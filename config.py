import dataclasses

import yaml
from typing import List
from dataclasses import dataclass

with open("config.yaml", "r") as __config_yaml:
    _config = yaml.load(__config_yaml, Loader=yaml.FullLoader)


@dataclass
class ReconstructionsConfig:
    inside_camera_parameters_path: str = _config["reconstruction"]["inside_camera_parameters_path"]
    thumb_connections: List[List[int]] = dataclasses.field(default_factory=_config["reconstruction"]["thumb_connections"])
    index_connections: List[List[int]] = dataclasses.field(default_factory=_config["reconstruction"]["index_connections"])
    middle_connections: List[List[int]] = dataclasses.field(default_factory=_config["reconstruction"]["middle_connections"])
    ring_connections: List[List[int]] = dataclasses.field(default_factory=_config["reconstruction"]["ring_connections"])
    pinkie_connections: List[List[int]] = dataclasses.field(default_factory=_config["reconstruction"]["pinkie_connections"])
    fingers_colors: List[List[str]] = dataclasses.field(default_factory=_config["reconstruction"]["fingers_colors"])
    world_image: int = _config["reconstruction"]["world_image"]


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
    ids: List[int] = dataclasses.field(default_factory=_config["camera"]["ids"])
    mode: str = _config["camera"]["mode"]
    width: int = _config["camera"]["width"]
    height: int = _config["camera"]["height"]


@dataclass
class FingersID:
    wrist: List[int] = dataclasses.field(default_factory=_config["fingers"]["wrist"])
    thumb: List[int] = dataclasses.field(default_factory=_config["fingers"]["thumb"])
    index: List[int] = dataclasses.field(default_factory=_config["fingers"]["index"])
    middle: List[int] = dataclasses.field(default_factory=_config["fingers"]["middle"])
    ring: List[int] = dataclasses.field(default_factory=_config["fingers"]["ring"])
    pinky: List[int] = dataclasses.field(default_factory=_config["fingers"]["pinky"])


@dataclass
class ArduinoArm:
    com: str = _config["arduino_arm"]["com"]
    baud: int = _config["arduino_arm"]["baud"]


ReconstructionsConfig.thumb_connections = _config["reconstruction"]["thumb_connections"]
ReconstructionsConfig.index_connections = _config["reconstruction"]["thumb_connections"]
ReconstructionsConfig.middle_connections = _config["reconstruction"]["index_connections"]
ReconstructionsConfig.ring_connections = _config["reconstruction"]["middle_connections"]
ReconstructionsConfig.pinkie_connections = _config["reconstruction"]["ring_connections"]
ReconstructionsConfig.fingers_colors = _config["reconstruction"]["pinkie_connections"]

CameraConfig.ids = _config["camera"]["ids"]

FingersID.wrist = _config["fingers"]["wrist"]
FingersID.thumb = _config["fingers"]["thumb"]
FingersID.index = _config["fingers"]["index"]
FingersID.middle = _config["fingers"]["middle"]
FingersID.ring = _config["fingers"]["ring"]
FingersID.pinky = _config["fingers"]["pinky"]