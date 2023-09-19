import os
import cv2
import json
import numpy as np
import random


def save_json(data: dict, path: str = "./", name_file: str = "Untitled"):
    with open(f"{path}/{name_file}.json", "w", encoding="UTF-8") as fp:
        fp.write(json.dumps(data))
        print(f"[!] File `{name_file}` saved successfully")


def read_json(path: str):
    with open(path, "r") as fp:
        data = json.load(fp)

    return data


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]

    A = np.array(A).reshape((4, 4))
    B = np.dot(A.transpose(), A)

    # B = A.transpose() @ A
    U, s, Vh = np.linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def triangulate(uvs1: np.ndarray, uvs2: np.ndarray,
                matrix1: np.ndarray, matrix2: np.ndarray,
                R: np.ndarray, T: np.ndarray):
    """_summary_

    Args:
        uvs1 (np.ndarray): _description_
        uvs2 (np.ndarray): _description_
        matrix1 (np.ndarray): _description_
        matrix2 (np.ndarray): _description_
        R (np.ndarray): _description_
        T (np.ndarray): _description_
    """

    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = matrix1 @ RT1

    RT2 = np.concatenate([R, T], axis=-1)
    P2 = matrix2 @ RT2

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)

    p3ds = np.array(p3ds)

    return p3ds


def mono_calibration_camera(path, rows, columns, resize=None, type_calib="chess", save=False):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

    imgpoints = []
    objpoints = []

    for fname in sorted(os.listdir(path)):
        img = cv2.imread(f"{path}/{fname}")
        if resize:
            img = cv2.resize(img, resize)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if type_calib == "circle":
            ret, corners = cv2.findCirclesGrid(gray, (rows, columns),
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        elif type_calib == "chess":
            ret, corners = cv2.findChessboardCorners(gray,
                                                     (rows, columns),
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        else:
            raise TypeError(f"It is not possible to calibrate the camera using the calibration type:: {type_calib}")

        if ret:
            conv_size = (3, 3)
            corners = cv2.cornerSubPix(gray, corners,
                                       conv_size, (-1, -1), criteria)

            img = cv2.drawChessboardCorners(img, (rows, columns),
                                            corners, ret)

            if save:
                cv2.imwrite(f"{path}/{fname}_drawChessboardCorners.png", img)

            objpoints.append(objp)
            imgpoints.append(corners)

        cv2.imshow("Calib", img)
        cv2.waitKey(10)

    cv2.destroyWindow("Calib")
    height, width = img.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       (width, height),
                                                       None, None)

    return ret, mtx, dist, rvecs, tvecs


def stereo_camera_calibration(matrix1: np.ndarray, dist1: np.ndarray,
                              matrix2: np.ndarray, dist2: np.ndarray,
                              frames_folder1: str, frames_folder2: str,
                              rows: int = 6, columns: int = 9, resize=None,
                              type_calib="chess", save=False):

    images1 = sorted(os.listdir(frames_folder1))
    images2 = sorted(os.listdir(frames_folder2))

    name_window1 = frames_folder1.split("/")[-1]
    name_window2 = frames_folder2.split("/")[-1]

    width = 0
    height = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

    stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC

    imgpoints_left = []
    imgpoints_right = []

    objpoints = []

    for img1, img2 in zip(images1, images2):
        frame1 = cv2.imread(f"{frames_folder1}/{img1}")
        frame2 = cv2.imread(f"{frames_folder2}/{img2}")

        if resize:
            frame1 = cv2.resize(frame1, resize)
            frame2 = cv2.resize(frame2, resize)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        height, width = frame1.shape[:2]

        if type_calib == "circle":
            ret1, corners1 = cv2.findCirclesGrid(gray1, (rows, columns),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret2, corners2 = cv2.findCirclesGrid(gray2, (rows, columns),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        elif type_calib == "chess":
            ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns),
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns),
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        else:
            raise TypeError(f"It is not possible to calibrate the camera using the calibration type:: {type_calib}")

        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (3, 3), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (3, 3), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, (rows, columns), corners1, ret1)
            cv2.drawChessboardCorners(frame2, (rows, columns), corners2, ret2)

            if save:
                cv2.imwrite(f"{frames_folder1}/{img1}_drawChessboardCorners.png", frame1)
                cv2.imwrite(f"{frames_folder2}/{img2}_drawChessboardCorners.png", frame2)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

        cv2.imshow(f"{name_window1}", frame1)
        cv2.imshow(f"{name_window2}", frame2)

        cv2.waitKey(500)

    ret, cm1, dist_c1, cm2, dist_c2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                                                      matrix1, dist1,
                                                                      matrix2, dist2, (width, height),
                                                                      criteria=criteria, flags=stereo_calibration_flags)

    # print(ret)
    cv2.destroyAllWindows()
    return R, T


def aruco_calibrate(path):

    counter, corners_list, id_list = [], [], []
    first = True
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()

    # TODO ПЕРЕПИСАТь
    arucoDict = cv2.aruco.getPredefinedDictionary(0)
    _PIXEL = 37.936267

    board = cv2.aruco.GridBoard((5, 7), .04, .02, arucoDict)

    for fname in sorted(os.listdir(path)):
        img = cv2.imread(f"{path}/{fname}")
        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,
                                                                  aruco_dict,
                                                                  parameters=arucoParams)

        if first:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))

    counter = np.array(counter)

    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(corners_list,
                                                                  id_list,
                                                                  counter,
                                                                  board,
                                                                  gray.shape,
                                                                  None, None)

    print("RMSE = ", ret)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)


def charuco_calibrate(path):
    """_summary_

    Args:
        path (_type_): _description_
    """
    counter, corners_list, id_list = [], [], []
    first = True
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()

    # TODO ПЕРЕПИСАТь
    arucoDict = cv2.aruco.getPredefinedDictionary(0)
    _PIXEL = 37.936267

    for fname in sorted(os.listdir(path)):
        img = cv2.imread(f"{path}/{fname}")
        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,
                                                                  aruco_dict,
                                                                  parameters=arucoParams)
        if first:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))

    counter = np.array(counter)

    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(corners_list,
                                                                    id_list,
                                                                    cv2.aruco.CharucoBoard((5, 7), .04, .02, arucoDict),
                                                                    gray.shape,
                                                                    None, None)

    print("RMSE = ", ret)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)


def get_world_space_origin(cmtx, dist, img_path,
                           rows: int = 6, columns: int = 9,
                           resize=None, type_calib="chess"):
    frame = cv2.imread(img_path, 1)

    if resize:
        frame = cv2.resize(frame, resize)

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if type_calib == "circle":
        ret, corners = cv2.findCirclesGrid(gray, (rows, columns), None)

    elif type_calib == "chess":
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    else:
        raise TypeError(f"It is not possible to calibrate the camera using the calibration type:: {type_calib}")

    cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
    cv2.putText(frame, "If you don't see detected points, try with a different image", (5, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('img', frame)
    cv2.waitKey(0)

    cv2.destroyWindow("img")

    ret, rvec, tvec = cv2.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec


def get_cam_to_world_transforms(cmtx0, dist0,
                                cmtx1, dist1,
                                R_W0, T_W0,
                                R_01, T_01,
                                image0,
                                image1):
    frame0 = cv2.imread(image0, 1)
    frame1 = cv2.imread(image1, 1)

    unitv_points = 5 * np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]],
                                dtype='float32').reshape((4, 1, 3))

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    points, _ = cv2.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame0, origin, _p, col, 2)

    # project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv2.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame1, origin, _p, col, 2)

    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    cv2.waitKey(0)

    return R_W1, T_W1
