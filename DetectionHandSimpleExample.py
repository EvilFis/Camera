import cv2
import mediapipe as mp

from config import FingersID, CameraConfig, DetectionConfig
from HandRecognition import draw_landmarks, calculate_angle, get_frame_keypoints


# ================== Инициализация значений ===============
fingers_info = {
    "Thumb": {
        "up": False,
        "angle": None,
        "id": FingersID.thumb
    },
    "Index": {
        "up": False,
        "angle": None,
        "id": FingersID.index
    },
    "Middle": {
        "up": False,
        "angle": None,
        "id": FingersID.middle
    },
    "Ring": {
        "up": False,
        "angle": None,
        "id": FingersID.ring
    },
    "Pinky": {
        "up": False,
        "angle": None,
        "id": FingersID.pinky
    }
}

mp_hands = mp.solutions.hands

# Настройки камеры
cap = cv2.VideoCapture(CameraConfig.ids[1])
cap.set(3, CameraConfig.width)
cap.set(4, CameraConfig.height)
# =========================================================

# ==================== Распознование ======================
with mp_hands.Hands(min_detection_confidence=DetectionConfig.detection_confidence,
                    min_tracking_confidence=DetectionConfig.tracking_confidence,
                    max_num_hands=DetectionConfig.num_hands) as hands:
    while cap.isOpened():

        key = cv2.waitKey(1) & 0xFF

        _, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        result = hands.process(image)
        image.flags.writeable = True

        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            hand_label = result.multi_handedness[0].classification[0].label
            key_points = get_frame_keypoints(landmarks=hand_landmarks,
                                             img=frame,
                                             count_points=len(mp_hands.HAND_CONNECTIONS))

            # Отрисовка линий на кадре
            draw_landmarks(frame, key_points, mp_hands.HAND_CONNECTIONS)

            for finger in fingers_info:
                angle = calculate_angle(
                    a=key_points[fingers_info[finger]["id"][0]],
                    b=key_points[fingers_info[finger]["id"][1]],
                    c=key_points[FingersID.wrist[0]]
                )

                fingers_info[finger]["angle"] = angle
                fingers_info[finger]["up"] = False

                if angle <= 90:
                    fingers_info[finger]["up"] = True

                if finger == "Thumb" and angle <= 130:
                    fingers_info[finger]["up"] = True

                cv2.putText(frame, str(round(angle, 2)),
                            key_points[fingers_info[finger]["id"][0]],
                            cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 255))

            for finger, i in zip(fingers_info, range(1, len(fingers_info) + 1)):
                cv2.putText(frame, f"{finger}: {fingers_info[finger]['up']}, {round(fingers_info[finger]['angle'], 2)}",
                            (20, 20 * i),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 0, 0))

        cv2.imshow("Original stream", frame)

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
