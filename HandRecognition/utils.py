import cv2
import numpy as np
from typing import NamedTuple


def get_frame_keypoints(landmarks: NamedTuple,
                        img: np.ndarray,
                        count_points: int) -> list:
    """
    Получение ключевых точек в реальных координатах
    :param landmarks: Распознанные ключевые точки
    :param img: Захваченный кадр
    :param count_points: Колличество точек
    :return: Список кортежей, в которых хранятся X и Y координат захваченного кадра
    """

    frame_keypoints = []

    for lm in landmarks:
        for point in range(count_points):
            pxl_x = int(round(img.shape[1] * lm.landmark[point].x))
            pxl_y = int(round(img.shape[0] * lm.landmark[point].y))
            frame_keypoints.append((pxl_x, pxl_y))

    return frame_keypoints


def draw_landmarks(img: np.ndarray,
                   landmark_list: list,
                   connection: list,
                   line_color: tuple = (255, 255, 255),
                   point_color: tuple = (0, 0, 255)
                   ) -> np.ndarray:
    """
    Отричсовка распознанных конечностей
    :param img: Захваченный кадр
    :param landmark_list: Список кардажей распознанных точек
    :param connection: Правила соединения линий
    :param line_color: Цвет линий
    :param point_color: Цвет точек
    :return:
    """

    if len(landmark_list) == 0:
        return

    if len(connection) != len(landmark_list):
        return

    for connect, center in zip(connection, landmark_list):
        img = cv2.line(img=img,
                       pt1=landmark_list[connect[0]],
                       pt2=landmark_list[connect[1]],
                       color=line_color,
                       thickness=2)

        img = cv2.circle(img=img,
                         center=center,
                         radius=3,
                         color=point_color,
                         thickness=cv2.FILLED)

    return img


def calculate_angle(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    """
    Расчёт угла наклона
    :param a: Точка А
    :param b: Точка Б
    :param c: Точка В
    :return: Угол наклона точки Б относительно точек А и В в градусах
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
