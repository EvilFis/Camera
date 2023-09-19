import numpy as np
from calibration_utils import mono_calibration_camera, save_json, read_json, stereo_camera_calibration


def auto_fill(dic: dict, data: list, names: list):

    info = dic.copy()

    for i, val in enumerate(names):
        info[val] = data[i]

    return info


# Camera calibration example
names = ["RMSE", "matrix", "dist", "rotVecs", "tvecs"]
camera_params = {}

# Calibration
print("Start rtsp calib")
ret1, mtx1, dist1, rvecs1, tvecs1 = mono_calibration_camera("E:/python_project/pythonProject/CameraTools/CONTENT/calib/mono/rtsp", 6, 9, resize=[640, 480], save=True)
rvecs1_1 = tuple(val.tolist() for val in rvecs1)
tvecs1_1 = tuple(val.tolist() for val in tvecs1)

print("Start d455 calib")
ret2, mtx2, dist2, rvecs2, tvecs2 = mono_calibration_camera("E:/python_project/pythonProject/CameraTools/CONTENT/calib/mono/Intel RealSense D455_151422254042/color", 
                                                            6, 9, resize=[640, 480], save=True)
rvecs2_1 = tuple(val.tolist() for val in rvecs2)
tvecs2_1 = tuple(val.tolist() for val in tvecs2)

print("Start l515 calib")
ret3, mtx3, dist3, rvecs3, tvecs3 = mono_calibration_camera("E:/python_project/pythonProject/CameraTools/CONTENT/calib/mono/Intel RealSense L515_f1230922/color", 
                                                            6, 9, resize=[640, 480], save=True)
rvecs3_1 = tuple(val.tolist() for val in rvecs3)
tvecs3_1 = tuple(val.tolist() for val in tvecs3)

# # Filling DICT format
camera1 = auto_fill(camera_params, [ret1, mtx1.tolist(), dist1.tolist(), rvecs1_1, tvecs1_1], names)
camera2 = auto_fill(camera_params, [ret2, mtx2.tolist(), dist2.tolist(), rvecs2_1, tvecs2_1], names)
camera3 = auto_fill(camera_params, [ret3, mtx3.tolist(), dist3.tolist(), rvecs3_1, tvecs3_1], names)

print("Save to files")

# Save to JSON
# save_json(camera1, path="E:/python_project/pythonProject/CameraTools/calibration", name_file="camera_rtsp")
# save_json(camera2, path="E:/python_project/pythonProject/CameraTools/calibration", name_file="camera_d455")
# save_json(camera3, path="E:/python_project/pythonProject/CameraTools/calibration", name_file="camera_lidar")


# load data to JSON
camera1 = read_json("E:/python_project/pythonProject/CameraTools/calibration/camera_rtsp.json")
camera2 = read_json("E:/python_project/pythonProject/CameraTools/calibration/camera_d455.json")
camera3 = read_json("E:/python_project/pythonProject/CameraTools/calibration/camera_lidar.json")

# Stereo camera calobration
names = ["R", "T"]

print("Calib rtsp + l515")
R, T = stereo_camera_calibration(np.array(camera1["matrix"]), np.array(camera1["dist"]), 
                                    np.array(camera3["matrix"]), np.array(camera3["dist"]), 
                                    "E:/python_project/pythonProject/CameraTools/CONTENT/calib/stereo/rtsp_lidar/Camera_RTSP",
                                    "E:/python_project/pythonProject/CameraTools/CONTENT/calib/stereo/rtsp_lidar/Intel RealSense L515_f1230922/color",
                                    resize=[640, 480], save=True)
stereo1 = auto_fill(camera_params, [R.tolist(), T.tolist()], names=names)
# save_json(stereo1, path="E:/python_project/pythonProject/CameraTools/calibration", name_file="camera_rtsp_l515")

print("Calib rtsp + d455")
R2, T2 = stereo_camera_calibration(np.array(camera1["matrix"]), np.array(camera1["dist"]), 
                                    np.array(camera2["matrix"]), np.array(camera2["dist"]), 
                                    "E:/python_project/pythonProject/CameraTools/CONTENT/calib/stereo/rtsp_d455/Camera_RTSP",
                                    "E:/python_project/pythonProject/CameraTools/CONTENT/calib/stereo/rtsp_d455/Intel RealSense D455_151422254042/color",
                                    resize=[640, 480], save=True)
stereo2 = auto_fill(camera_params, [R2.tolist(), T2.tolist()], names=names)
# save_json(stereo2, path="E:/python_project/pythonProject/CameraTools/calibration", name_file="camera_rtsp_d455")

print("Calib l515 + d455")
R3, T3 = stereo_camera_calibration(np.array(camera3["matrix"]), np.array(camera3["dist"]), 
                                    np.array(camera2["matrix"]), np.array(camera2["dist"]), 
                                    "E:/python_project/pythonProject/CameraTools/CONTENT/calib/stereo/lidar_d455/Intel RealSense L515_f1230922/color",
                                    "E:/python_project/pythonProject/CameraTools/CONTENT/calib/stereo/lidar_d455/Intel RealSense D455_151422254042/color",
                                    resize=[640, 480], save=True)
stereo3 = auto_fill(camera_params, [R3.tolist(), T3.tolist()], names=names)
# save_json(stereo3, path="E:/python_project/pythonProject/CameraTools/calibration", name_file="camera_l515_d455")