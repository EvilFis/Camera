import yaml
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    detection_confidence: float
    tracking_confidence: float
    num_hands: int


@dataclass
class CameraConfig:
    ids: list[int]
    mode: str
    width: int
    height: int


@dataclass
class FingersID:
    wrist: int
    thumb: list[int]
    index: list[int]
    middle: list[int]
    ring: list[int]
    pinky: list[int]


@dataclass
class Config:
    detection: DetectionConfig
    camera: CameraConfig
    fingersID: FingersID


with open("config.yaml", "r") as __config_yaml:
    __config = yaml.load(__config_yaml, Loader=yaml.FullLoader)

__detection_conf = DetectionConfig(
    detection_confidence=__config["detection"]["min_detection_confidence"],
    tracking_confidence=__config["detection"]["min_tracking_confidence"],
    num_hands=__config["detection"]["max_num_hands"]
)

__camera_config = CameraConfig(
    ids=__config["camera"]["ids"],
    mode=__config["camera"]["mode"],
    width=__config["camera"]["width"],
    height=__config["camera"]["height"]
)

__fingersID_config = FingersID(
    wrist=__config["fingers"]["wrist"],
    thumb=__config["fingers"]["thumb"],
    index=__config["fingers"]["index"],
    middle=__config["fingers"]["middle"],
    ring=__config["fingers"]["ring"],
    pinky=__config["fingers"]["pinky"]
)

config = Config(detection=__detection_conf,
                camera=__camera_config,
                fingersID=__fingersID_config)
