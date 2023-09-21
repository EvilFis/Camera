import cv2
import queue
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Barrier, Pipe

from Camera import Camera
from Calibration import read_json
from HandRecognition import get_frame_keypoints, draw_landmarks, calculate_angle
from config import FingersID, CameraConfig, DetectionConfig, ReconstructionsConfig


def _make_homogeneous_rep_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P


def __prediction(image: np.ndarray,
                 model: mp.solutions.hands.Hands,
                 fingers_info: dict,
                 ) -> tuple[np.ndarray, dict, list[tuple[int, int]]]:
    """
    Функция обертка для обобщения процесса детекции объекта
    :param image: Захваченный кадр
    :param model: Модель распознования
    :param fingers_info: Информация о пальцах
    :return: Захваченный кадр и информацию о пальцах
    """

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb_frame.flags.writeable = False
    prediction = model.process(rgb_frame)
    rgb_frame.flags.writeable = True

    hand_landmarks = prediction.multi_hand_landmarks

    key_points = [(-1, -1) for _ in range(len(mp.solutions.hands.HAND_CONNECTIONS))]

    if hand_landmarks:
        # hand_label = prediction.multi_handedness[0].classification[0].label
        key_points = get_frame_keypoints(landmarks=hand_landmarks,
                                         img=image,
                                         count_points=len(mp.solutions.hands.HAND_CONNECTIONS))

        image = draw_landmarks(image, key_points, mp.solutions.hands.HAND_CONNECTIONS)

        for finger in fingers_info:
            angle = calculate_angle(
                a=key_points[fingers_info[finger]["id"][0]],
                b=key_points[fingers_info[finger]["id"][1]],
                c=key_points[FingersID.wrist[0]]
            )

            fingers_info[finger]["angle"] = angle
            fingers_info[finger]["close_finger"] = False

            if angle <= 90:
                fingers_info[finger]["close_finger"] = True

            if finger == "Thumb" and angle <= 135:
                fingers_info[finger]["close_finger"] = True

            cv2.putText(image, str(round(angle, 2)),
                        key_points[fingers_info[finger]["id"][0]],
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 255))

        for finger, i in zip(fingers_info, range(1, len(fingers_info) + 1)):
            cv2.putText(image,
                        f"{finger}: {fingers_info[finger]['close_finger']}, {round(fingers_info[finger]['angle'], 2)}",
                        (20, 20 * i),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 0))

    return image, fingers_info, key_points


def DLT(P1: np.ndarray, P2: np.ndarray, point1: list, point2: list) -> np.ndarray:
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]

    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = np.linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def visualize_3d(barrier: Barrier, receiver: Pipe):

    barrier.wait()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Примените координатные повороты к точке по оси z как вверх
    Rz = np.array(([[0., -1., 0.],
                    [1., 0., 0.],
                    [0., 0., 1.]]))

    Rx = np.array(([[1., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., -1.]]))

    thumb_f = [[0, 1], [1, 2], [2, 3], [3, 4]]
    index_f = [[0, 5], [5, 6], [6, 7], [7, 8]]
    middle_f = [[0, 9], [9, 10], [10, 11], [11, 12]]
    ring_f = [[0, 13], [13, 14], [14, 15], [15, 16]]
    pinkie_f = [[0, 17], [17, 18], [18, 19], [19, 20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

    while True:
        key_points = receiver.recv()

        p3ds_rotated = []

        for kpt in key_points:
            kpt_rotated = Rz @ Rx @ kpt
            p3ds_rotated.append(kpt_rotated)
            # p3ds_rotated.append(kpt)

        p3ds_rotated = np.array(p3ds_rotated)

        for finger, finger_color in zip(fingers, fingers_colors):
            for _c in finger:
                ax.plot(xs=[p3ds_rotated[_c[0], 0], p3ds_rotated[_c[1], 0]],
                        ys=[p3ds_rotated[_c[0], 1], p3ds_rotated[_c[1], 1]],
                        zs=[p3ds_rotated[_c[0], 2], p3ds_rotated[_c[1], 2]],
                        linewidth=4, c=finger_color)

        # draw axes
        ax.plot(xs=[0, 5], ys=[0, 0], zs=[0, 0], linewidth=2, color='red')
        ax.plot(xs=[0, 0], ys=[0, 5], zs=[0, 0], linewidth=2, color='blue')
        ax.plot(xs=[0, 0], ys=[0, 0], zs=[0, 5], linewidth=2, color='black')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylim3d(-10, 10)
        ax.set_ylabel('y')
        ax.set_zlim3d(-15, 10)
        ax.set_zlabel('z')

        ax.elev = 20
        ax.azim = 145
        plt.pause(0.001)
        ax.cla()


def hand_detection(barrier: Barrier,
                   receiver_left_cam: Pipe = None,
                   receiver_right_cam: Pipe = None,
                   send_visualize: Pipe = None,
                   inside_cameras_parameters: dict = None):
    """
    Обобщенная функция по расспознованию ключевых точек на руке
    :param send_visualize:
    :param inside_cameras_parameters:
    :param barrier: Ожидание процессов
    :param receiver_left_cam: Полученние данных с камеры 1
    :param receiver_right_cam: Получение данных с камеры 2
    :return:
    """

    if inside_cameras_parameters is None:
        inside_cameras_parameters = {}

    fingers_info_left_camera = {
        "Thumb": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.thumb
        },
        "Index": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.index
        },
        "Middle": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.middle
        },
        "Ring": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.ring
        },
        "Pinky": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.pinky
        }
    }
    fingers_info_right_camera = {
        "Thumb": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.thumb
        },
        "Index": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.index
        },
        "Middle": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.middle
        },
        "Ring": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.ring
        },
        "Pinky": {
            "close_finger": False,
            "angle": None,
            "id": FingersID.pinky
        }
    }

    barrier.wait()

    hands_left = mp.solutions.hands.Hands(min_detection_confidence=DetectionConfig.detection_confidence,
                                          min_tracking_confidence=DetectionConfig.tracking_confidence,
                                          max_num_hands=DetectionConfig.num_hands)

    hands_right = mp.solutions.hands.Hands(min_detection_confidence=DetectionConfig.detection_confidence,
                                           min_tracking_confidence=DetectionConfig.tracking_confidence,
                                           max_num_hands=DetectionConfig.num_hands)

    left_matrix = np.array(inside_cameras_parameters["camera_left"]["matrix"])
    right_matrix = np.array(inside_cameras_parameters["camera_right"]["matrix"])
    R = np.array(inside_cameras_parameters["R"])
    T = np.array(inside_cameras_parameters["T"])

    RT1 = np.eye(3, 4)
    # left_R = np.array(inside_cameras_parameters["camera_left"]["rotVecs"][1])
    # left_T = np.array(inside_cameras_parameters["camera_left"]["tvecs"][1])

    # RT1 = np.concatenate([left_R, left_T], axis=-1)
    P0 = left_matrix @ RT1

    RT2 = np.concatenate([R, T], axis=-1)
    P1 = right_matrix @ RT2

    # P0 = left_matrix @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    # P1 = right_matrix @ _make_homogeneous_rep_matrix(R, T)[:3, :]

    while True:
        key = cv2.waitKey(1) & 0xFF

        frame_left_camera = receiver_left_cam.recv()
        frame_right_camera = receiver_right_cam.recv()

        frame_left_camera, fingers_info_left_camera, kps_left = __prediction(frame_left_camera, hands_left,
                                                                             fingers_info_left_camera)

        frame_right_camera, fingers_info_right_camera, kps_right = __prediction(frame_right_camera, hands_right,
                                                                                fingers_info_right_camera)

        frame_p3ds = []
        for uv1, uv2 in zip(kps_left, kps_right):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((len(mp.solutions.hands.HAND_CONNECTIONS), 3))
        print(frame_p3ds)
        send_visualize.send(frame_p3ds)

        cv2.imshow("Camera Left", frame_left_camera)
        cv2.imshow("Camera Right", frame_right_camera)

        if key == ord('q'):
            cv2.destroyAllWindows()
            hands_left.close()
            hands_right.close()
            break


def main():
    web_cam1 = Camera(device_id=CameraConfig.ids[0],
                      mode=CameraConfig.mode,
                      width=CameraConfig.width,
                      height=CameraConfig.height)

    web_cam2 = Camera(device_id=CameraConfig.ids[1],
                      mode=CameraConfig.mode,
                      width=CameraConfig.width,
                      height=CameraConfig.height)

    inside_stereo_parameters = read_json(f"{ReconstructionsConfig.inside_camera_parameters_path}/stereo_params.json")

    barrier = Barrier(4)
    rec_web_cam1, send_web_cam1 = Pipe()
    rec_web_cam2, send_web_cam2 = Pipe()
    rec_visualize, send_visualize = Pipe()

    cam1_args = {
        "barrier": barrier,
        "show_gui": False,
        "sender": send_web_cam1,
    }

    cam2_args = {
        "barrier": barrier,
        "show_gui": False,
        "sender": send_web_cam2,
    }

    mp_web1 = Process(target=web_cam1.stream, kwargs=cam1_args, name="CameraLeft")
    mp_web2 = Process(target=web_cam2.stream, kwargs=cam2_args, name="CameraRight")
    visualize = Process(target=visualize_3d, args=(barrier, rec_visualize,), name="Visualize 3D")
    res_cam = Process(target=hand_detection,
                      args=(barrier, rec_web_cam1, rec_web_cam2, send_visualize, inside_stereo_parameters,),
                      name="HandDetection")

    mp_web1.start()
    mp_web2.start()
    visualize.start()
    res_cam.start()

    res_cam.join()

    mp_web1.terminate()
    mp_web2.terminate()
    visualize.terminate()


if __name__ == "__main__":
    main()
