# python mono_calibration.py --stream-off - Запуск каллибровки без запуска камеры
# python mono_calibration.py - Запуск калибровки с запуском камеры

import os
import shutil
import argparse

from Camera import Camera
from config import CalibrationConfig, ReconstructionsConfig, CameraConfig
from Calibration import mono_calibration_camera, save_json

parser = argparse.ArgumentParser(description="Test file")
parser.add_argument("-so", "--stream-off", dest="stream",
                    type=bool, default=True, const=False, nargs="?", help="Using the camera (default True)")

arguments = parser.parse_args()

if arguments.stream:
    try:
        shutil.rmtree(CalibrationConfig.mono_calibration_path)
    except FileNotFoundError:
        pass

    try:
        os.mkdir(CalibrationConfig.mono_calibration_path)
        os.mkdir(ReconstructionsConfig.inside_camera_parameters_path)
    except FileExistsError:
        pass


for camera_id in CameraConfig.ids:

    if arguments.stream:
        camera = Camera(device_id=camera_id,
                        mode="frame",
                        width=CameraConfig.width,
                        height=CameraConfig.height)

        camera.stream(img_count=CalibrationConfig.img_count,
                      time_out=CalibrationConfig.time_out,
                      path=CalibrationConfig.mono_calibration_path,
                      show_gui=CalibrationConfig.show_gui)

    ret, mtx, dist, rvecs, tvecs = mono_calibration_camera(
        path=f"{CalibrationConfig.mono_calibration_path}/Camera_{camera_id}_frame/",
        rows=CalibrationConfig.rows,
        columns=CalibrationConfig.columns,
        save=CalibrationConfig.save
    )

    rvecs = tuple(val.tolist() for val in rvecs)
    tvecs = tuple(val.tolist() for val in tvecs)

    inside_camera_params = {
        "RMSE": ret,
        "matrix": mtx.tolist(),
        "dist": dist.tolist(),
        "rotVecs": rvecs,
        "tvecs": tvecs
    }

    save_json(data=inside_camera_params,
              path=ReconstructionsConfig.inside_camera_parameters_path,
              name_file=f"camera_{camera_id}")


print("[!] Монокалибровка камер прошла успешно")