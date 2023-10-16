import cv2
from Camera import Camera
from multiprocessing import Process, Barrier, Pipe, Value

modes = ("stream", "video", "frame")
mode = modes[0]

web_cam1 = Camera(device_id=0, mode=mode)
web_cam2 = Camera(device_id=1, mode=mode)
rec_web_cam1, send_web_cam1 = Pipe()
rec_web_cam2, send_web_cam2 = Pipe()


def show_result(barrier: Barrier, receiver_cam1: Pipe = None, receiver_cam2: Pipe = None):

    barrier.wait()

    while True:
        key = cv2.waitKey(1)

        if receiver_cam1 and receiver_cam2:
            frame_cam1 = receiver_cam1.recv()
            frame_cam2 = receiver_cam2.recv()

            cv2.imshow("Camera 0.1", frame_cam1)
            cv2.imshow("Camera 1.1", frame_cam2)

            if key == ord('q') & 0xFF:
                cv2.destroyAllWindows()
                break


def main() -> None:
    barrier = Barrier(3)

    web_args_1 = {
        "fps": 30,
        "barrier": barrier,
        "show_gui": False,
        "sender": send_web_cam1,
    }

    web_args_2 = {
        "fps": 30,
        "barrier": barrier,
        "show_gui": False,
        "sender": send_web_cam2,
    }

    multiprocessing_web1 = Process(target=web_cam1.stream, kwargs=web_args_1, name="Camera 1")
    multiprocessing_web2 = Process(target=web_cam2.stream, kwargs=web_args_2, name="Camera 2")
    result_cameras = Process(target=show_result, args=(barrier, rec_web_cam1, rec_web_cam2, ),  name="result cameras")

    multiprocessing_web1.start()
    multiprocessing_web2.start()
    result_cameras.start()

    result_cameras.join()
    multiprocessing_web1.terminate()
    multiprocessing_web2.terminate()





if __name__ == "__main__":
    main()