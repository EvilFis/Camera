import yaml
from typing import ClassVar
from dataclasses import dataclass


with open("config.yaml", "r") as __config_yaml:
    _config = yaml.load(__config_yaml, Loader=yaml.FullLoader)


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
