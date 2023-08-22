from Camera import Camera, CameraRS # Самописные библиотеки
from multiprocessing import Barrier, Process

if __name__ == "__main__":
    
    mods = ("stream", "video", "frame")
    mode = mods[1]
    
    print()
    print("Доступные устройства камеры RealSense:")
    for device in CameraRS.get_devices_str():
        print("     ", device)
    print()
    
    rtsp = Camera(device_id="rtsp://grant:QWERTYUIOP{}@192.168.18.25:554/cam/realmonitor?channel=1&subtype=1", mode=mode)
    D455 = CameraRS(device_id=0, mode=mode)
    L515 = CameraRS(device_id=1, mode=mode)
    
    print(D455)
    print(L515)
    
    print()
    print(f"Доступные сенсоры камеры {L515.get_device_name()} #{L515.get_serial_number}:")
    sensors = D455.get_sensors()
    for sensor in sensors:
        print("     ", sensor)
        
    print()
    print(f"Выбранный сенсор {sensors[0]} имеет следующие профили:")
    for profile in L515.get_profiles(0):
        print("     ", profile)
    
    print()
    print("#########################################################")
    print()
    print(f"Выбранный сенсор {sensors[1]} имеет следующие профили:")
    for profile in L515.get_profiles(1):
        print("     ", profile)
    
    barrier = Barrier(3)
    
    args_d455 = {
        "color_profile": 96,
        "depth_profile": 226,
        "show_gui_color": True,
        "show_gui_depth": False,
        "img_count": 3,
        "time_out": 3,
        "barrier": barrier,
    }
    
    args_l515 = {
        "color_profile": 89,
        "depth_profile": 7,
        "show_gui_color": True,
        "show_gui_depth": False,
        "img_count": 3,
        "time_out": 3,
        "barrier": barrier,
    }
    
    args_rtsp = {
        "img_count": 3,
        "time_out": 3,
        "fps": 30,
        "barrier": barrier,
    }
    
    rtsp_proc = Process(target=rtsp.stream, kwargs=args_rtsp, name="RTSP")
    D455_proc = Process(target=D455.stream, kwargs=args_d455, 
                        name=f"{D455.get_device_name()} #{D455.get_serial_number}")
    L515_proc = Process(target=L515.stream, kwargs=args_l515, 
                        name=f"{L515.get_device_name()} #{L515.get_serial_number}")
    rtsp_proc.start()
    D455_proc.start()
    L515_proc.start()
    