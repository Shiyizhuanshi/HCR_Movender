#!/usr/bin/env python3
import threading
import select
from evdev import InputDevice, ecodes, list_devices

class GamepadController:
    def __init__(self, device_path='/dev/input/event6'):
        """
        初始化控制器，device_path 请根据实际情况修改。
        """
        self.device_path = device_path
        self.gamepad = InputDevice(device_path)
        try:
            # 获取 ABS_X 与 ABS_Y 的校准信息
            self.abs_x_info = self.gamepad.absinfo(ecodes.ABS_X)
            self.abs_y_info = self.gamepad.absinfo(ecodes.ABS_Y)
        except Exception as e:
            print("获取轴校准信息失败:", e)
            self.abs_x_info = None
            self.abs_y_info = None
        self.running = False
        self.listener_thread = None

    def map_axis(self, raw, absinfo):
        """
        更准确地将轴原始数据映射到 -255 到 255 的范围。
        """
        if absinfo is None:
            return 0

        # 直接使用设备实际报告的min/max范围
        min_val = absinfo.min
        max_val = absinfo.max
        
        # 确保不会除以零
        if max_val == min_val:
            return 0

        # 线性映射到-255 ~ 255
        mapped = ((raw - min_val) / (max_val - min_val)) * 510 - 255

        # 限制映射后的值范围
        mapped = max(min(mapped, 255), -255)

        return int(mapped)


    def _event_loop(self, callback):
        """
        后台线程事件循环，等待设备事件，并调用回调函数。
        回调函数格式：callback(event_type, data)
         - 对于 ABS_X 与 ABS_Y 事件，data 为映射后的值。
         - 对于其他轴事件，直接返回原始值。
         - 对于按键事件，data 为 (key_name, state) 的元组，state 可能为 "Pressed"、"Released" 或 "Held"。
        """
        while self.running:
            # 使用 select 进行超时等待，避免阻塞无法退出
            r, _, _ = select.select([self.gamepad], [], [], 1)
            if not r:
                continue
            for event in self.gamepad.read():
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X and self.abs_x_info is not None:
                        mapped_x = self.map_axis(event.value, self.abs_x_info)
                        callback('ABS_X', mapped_x)
                    elif event.code == ecodes.ABS_Y and self.abs_y_info is not None:
                        mapped_y = self.map_axis(event.value, self.abs_y_info)
                        callback('ABS_Y', mapped_y)
                    else:
                        # 对于其他绝对轴事件，直接返回原始数值
                        callback(f"ABS_{event.code}", event.value)
                elif event.type == ecodes.EV_KEY:
                    # 按键事件状态：0-Released, 1-Pressed, 2-Held
                    if event.value == 1:
                        state = "Pressed"
                    elif event.value == 0:
                        state = "Released"
                    elif event.value == 2:
                        state = "Held"
                    else:
                        state = f"value={event.value}"
                    try:
                        key_name = ecodes.KEY[event.code]
                    except KeyError:
                        key_name = f"Unknown({event.code})"
                    callback('KEY', (key_name, state))

    def register_listener(self, callback):
        """
        注册监听事件，启动后台线程。
        :param callback: 回调函数，格式为 callback(event_type, data)
        """
        if self.running:
            raise Exception("Listener already running.")
        self.running = True
        self.listener_thread = threading.Thread(target=self._event_loop, args=(callback,))
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def unregister_listener(self):
        """
        注销监听事件，停止后台线程。
        """
        self.running = False
        if self.listener_thread:
            self.listener_thread.join()
            self.listener_thread = None

    def read_device_input(self):
        """
        阻塞读取一次设备输入，返回一个事件对象。
        适用于同步读取单个事件的场景。
        """
        r, _, _ = select.select([self.gamepad], [], [])
        for event in self.gamepad.read():
            return event
        return None

# 以下代码用于模块自身测试，可在直接运行该模块时使用
if __name__ == "__main__":
    devices = [InputDevice(path) for path in list_devices()]
    print("可用的输入设备：")
    for i, device in enumerate(devices):
        print(f"[{i}] {device.path}: {device.name}")

    idx = int(input("请选择需要监听的设备索引："))
    selected_device = devices[idx]

    def my_callback(event_type, data):
        print(f"Event: {event_type}, Data: {data}")

    controller = GamepadController(selected_device.path)
    print("正在监听设备：", controller.gamepad.name)
    controller.register_listener(my_callback)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        controller.unregister_listener()
        print("监听已停止")
