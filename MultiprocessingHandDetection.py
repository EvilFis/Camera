import cv2
import numpy as np
import mediapipe as mp

from multiprocessing import Process, Barrier, Pipe

from Camera import Camera
from config import FingersID, CameraConfig, DetectionConfig
from HandRecognition import calculate_angle, draw_landmarks, get_frame_keypoints


def __prediction(image: np.ndarray,
                 model: mp.solutions.hands.Hands,
                 fingers_info: dict) -> tuple[np.ndarray, dict]:

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

        print(key_points) # TODO УБрать

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

    return image, fingers_info


def hand_detection(barrier: Barrier,
                   receiver_left_cam: Pipe = None,
                   receiver_right_cam: Pipe = None):

    """
    Обобщенная функция по расспознованию ключевых точек на руке
    :param barrier: Ожидание процессов
    :param receiver_left_cam: Полученние данных с камеры 1
    :param receiver_right_cam: Получение данных с камеры 2
    :return:
    """

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

    while True:
        key = cv2.waitKey(1) & 0xFF

        frame_left_camera = receiver_left_cam.recv()
        frame_right_camera = receiver_right_cam.recv()

        frame_left_camera, fingers_info_left_camera = __prediction(frame_left_camera, hands_left,
                                                                   fingers_info_left_camera)

        frame_right_camera, fingers_info_right_camera = __prediction(frame_right_camera, hands_right,
                                                                     fingers_info_right_camera)

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

    barrier = Barrier(3)
    rec_web_cam1, send_web_cam1 = Pipe()
    rec_web_cam2, send_web_cam2 = Pipe()

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
    res_cam = Process(target=hand_detection,
                      args=(barrier, rec_web_cam1, rec_web_cam2,),
                      name="HandDetection")

    mp_web1.start()
    mp_web2.start()
    res_cam.start()

    res_cam.join()

    mp_web1.terminate()
    mp_web2.terminate()


if __name__ == "__main__":
    main()
