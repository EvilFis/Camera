import cv2
import time
import serial
import threading
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Barrier, Pipe


from Camera import Camera
from Calibration import read_json
from HandRecognition import get_frame_keypoints, draw_landmarks, calculate_angle
from config import FingersID, CameraConfig, DetectionConfig, ReconstructionsConfig, ArduinoArm


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


def DLT(P1: np.ndarray, P2: np.ndarray,
        point1: list, point2: list) -> np.ndarray:

    """
    Прямые линейные преобразования
    :param P1: Матрица проекций 1
    :param P2: Матрица проекций 2
    :param point1: Ключевая точка камеры 1
    :param point2: Ключевая точка камеры 2
    :return: вектор 3D точек
    """

    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]

    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = np.linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def visualize_3d(barrier: Barrier,
                 receiver: Pipe):

    """
    Функция визуализации триангулируемых данных
    :param barrier: Барьер для ожидания других потоков
    :param receiver: Получение данных с потока
    :return:
    """

    barrier.wait()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.elev = 35
    ax.azim = 70

    # Примените координатные повороты к точке по оси z как вверх
    Rz = np.array(([[0., -1., 0.],
                    [1., 0., 0.],
                    [0., 0., 1.]]))

    Rx = np.array(([[1., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., -1.]]))

    # Информация по правильности построения пальцев представлена в config.yaml
    fingers = [ReconstructionsConfig.pinkie_connections,
               ReconstructionsConfig.ring_connections,
               ReconstructionsConfig.middle_connections,
               ReconstructionsConfig.index_connections,
               ReconstructionsConfig.thumb_connections]

    fingers_colors = ReconstructionsConfig.fingers_colors

    while True:
        key_points = receiver.recv()

        p3ds_rotated = []

        # Поворо точек относительно коордиант
        for kpt in key_points:
            kpt_rotated = Rz @ Rx @ kpt
            p3ds_rotated.append(kpt_rotated)

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

        ax.set_xlim3d(-7, 7)
        ax.set_xlabel('x')
        ax.set_ylim3d(-7, 7)
        ax.set_ylabel('y')
        ax.set_zlim3d(-10, 10)
        ax.set_zlabel('z')

        plt.pause(0.0001)
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

    # Информация о пальцах получаемых с камер
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

    # Ожидаем подключение всех паралельных процессов
    barrier.wait()

    # Инициализация базовых параметров распознования рук
    hands_left = mp.solutions.hands.Hands(min_detection_confidence=DetectionConfig.detection_confidence,
                                          min_tracking_confidence=DetectionConfig.tracking_confidence,
                                          max_num_hands=DetectionConfig.num_hands)

    hands_right = mp.solutions.hands.Hands(min_detection_confidence=DetectionConfig.detection_confidence,
                                           min_tracking_confidence=DetectionConfig.tracking_confidence,
                                           max_num_hands=DetectionConfig.num_hands)

    # Расчёт триангуляции
    left_matrix = np.array(inside_cameras_parameters["camera_left"]["matrix"])
    right_matrix = np.array(inside_cameras_parameters["camera_right"]["matrix"])
    R = np.array(inside_cameras_parameters["R"])
    T = np.array(inside_cameras_parameters["T"])

    # для триангуляции берем вектор поворотов радригеса с откалиброванного кадра (выбираем сами), а также вектор
    # трансформаций
    left_R = np.array(inside_cameras_parameters["camera_left"]["rotVecs"])[ReconstructionsConfig.world_image]
    left_T = np.array(inside_cameras_parameters["camera_left"]["tvecs"])[ReconstructionsConfig.world_image]

    # Получение вторых параметров
    left_R, _ = cv2.Rodrigues(left_R)
    right_R = R @ left_R
    right_T = R @ left_T + T

    # Расчёт матриц проекций
    P0 = left_matrix @ np.concatenate([left_R, left_T], axis=-1)
    P1 = right_matrix @ np.concatenate([right_R, right_T], axis=-1)

    arduino = serial.Serial(port=ArduinoArm.com, baudrate=ArduinoArm.baud)
    arduino_code = 200

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Получение данных с паралельных потоков камер
        frame_left_camera = receiver_left_cam.recv()
        frame_right_camera = receiver_right_cam.recv()

        #  предсказание с дальнешим отображением данных
        frame_left_camera, fingers_info_left_camera, kps_left = __prediction(frame_left_camera, hands_left,
                                                                             fingers_info_left_camera)

        frame_right_camera, fingers_info_right_camera, kps_right = __prediction(frame_right_camera, hands_right,
                                                                                fingers_info_right_camera)

        # Установка общего значения пальцев
        fingers_info = {
            "Thumb": fingers_info_left_camera["Thumb"]["close_finger"] or fingers_info_right_camera["Thumb"]["close_finger"],
            "Index": fingers_info_left_camera["Index"]["close_finger"] or fingers_info_right_camera["Index"]["close_finger"],
            "Middle": fingers_info_left_camera["Middle"]["close_finger"] or fingers_info_right_camera["Middle"]["close_finger"],
            "Ring": fingers_info_left_camera["Ring"]["close_finger"] or fingers_info_right_camera["Ring"]["close_finger"],
            "Pinky": fingers_info_left_camera["Pinky"]["close_finger"] or fingers_info_right_camera["Pinky"]["close_finger"]
        }

        # Производим прямые линейные преобразования
        frame_p3ds = []
        for uv1, uv2 in zip(kps_left, kps_right):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((len(mp.solutions.hands.HAND_CONNECTIONS), 3))

        message = "$"
        for i in fingers_info:
            message += str(int(not fingers_info[i]))
        message += "$"

        if arduino_code == 200:
            arduino.write(bytes(message, "utf-8"))

        arduino_code = int(arduino.read(3).decode())

        # Визуализируем полученные результаты
        send_visualize.send(frame_p3ds)

        # Отображаем кадры с отрисовкой ключевых точек
        cv2.imshow("Camera Left", frame_left_camera)
        cv2.imshow("Camera Right", frame_right_camera)

        if key == ord('q'):
            cv2.destroyAllWindows()
            hands_left.close()
            hands_right.close()
            break


def main():
    """
    Основная функция, которая запускает поток камер + отрисовку распознанных точек + 3D реконструкцию
    :return:
    """
    web_cam1 = Camera(device_id=CameraConfig.ids[0],
                      mode=CameraConfig.mode,
                      width=CameraConfig.width,
                      height=CameraConfig.height)

    web_cam2 = Camera(device_id=CameraConfig.ids[1],
                      mode=CameraConfig.mode,
                      width=CameraConfig.width,
                      height=CameraConfig.height)

    # Загрузка информации о стерео параметрах камеры
    inside_stereo_parameters = read_json(f"{ReconstructionsConfig.inside_camera_parameters_path}/stereo_params.json")

    # Устанавлием барьер, который синхронихирует процессы между собой (дождется запуска 4 процессов)
    barrier = Barrier(4)

    # Устанавливаем общение между потоками
    rec_web_cam1, send_web_cam1 = Pipe()  # Общение между камерой 1 и отрисовкой контрольных точек
    rec_web_cam2, send_web_cam2 = Pipe()  # Общение между камерой 2 и отрисовкой контрольных точек
    rec_visualize, send_visualize = Pipe()  # Общение между отрисовкой контрольных точек и 3D реконструкцией

    # Параметры камеры 1 (Информация в классе самой камеры)
    cam1_args = {
        "barrier": barrier,
        "show_gui": False,
        "sender": send_web_cam1,
    }

    # Параметры камеры 2 (Информация в классе самой камеры)
    cam2_args = {
        "barrier": barrier,
        "show_gui": False,
        "sender": send_web_cam2,
    }

    # Объявление процессов
    mp_web1 = Process(target=web_cam1.stream, kwargs=cam1_args, name="CameraLeft")
    mp_web2 = Process(target=web_cam2.stream, kwargs=cam2_args, name="CameraRight")
    visualize = Process(target=visualize_3d, args=(barrier, rec_visualize,), name="Visualize 3D")
    res_cam = Process(target=hand_detection,
                      args=(barrier, rec_web_cam1, rec_web_cam2,
                            send_visualize, inside_stereo_parameters,),
                      name="HandDetection")

    #  Запуск процессов
    mp_web1.start()
    mp_web2.start()
    visualize.start()
    res_cam.start()

    # Дожидаемся завершения работы отрисовки ключевых точек
    res_cam.join()

    # Убиваем остальные процессыт
    mp_web1.terminate()
    mp_web2.terminate()
    visualize.terminate()


if __name__ == "__main__":
    main()
