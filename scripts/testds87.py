import sys
import os

SCEPTER_SDK_ROOT = "/home/a104/ScepterSDK"
DEFAULT_DS87_PY_PATH = os.path.join(SCEPTER_SDK_ROOT, "MultilanguageSDK", "Python")

if DEFAULT_DS87_PY_PATH in sys.path:
    sys.path.remove(DEFAULT_DS87_PY_PATH)
sys.path.insert(0, DEFAULT_DS87_PY_PATH)

from API import ScepterDS_api as sdk

cam = sdk.ScepterTofCam()

camera_count = cam.scGetDeviceCount(3000)
print("camera_count =", camera_count)

if camera_count > 0:
    ret, device_array = cam.scGetDeviceInfoList(camera_count)
    print("ret =", ret)
    print("device_array =", device_array)
    print("len(device_array) =", len(device_array))

    for i in range(len(device_array)):
        dev = device_array[i]
        print("=" * 40)
        print("index =", i)
        print("type(dev) =", type(dev))
        print("dir(dev) =", dir(dev))
        print("dev =", dev)

        # 逐个试字段
        for key in ["serialNumber", "ip", "status", "mac", "cameraName", "deviceType"]:
            try:
                value = getattr(dev, key)
                print(key, "=", value)
            except Exception as e:
                print(key, "read failed:", e)