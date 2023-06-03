
# 如果系统支持，可以通过禁用和启用设备来模拟热插拔。 
# 例如，如果使用的是带 PCI 插槽的系统，则可以使用 `echo 1 > /sys/bus/pci/devices/DEVICE_ID/remove` 命令来模拟移除设备
# `echo 1 > /sys/bus /pci/rescan 模拟添加设备的命令。
import pytest
import subprocess

def hotplug_nvidia():
    # Get the device ID for the Nvidia GPU...
    device_id = get_nvidia_device_id()
    try:
        # Remove the device
        with open(f"/sys/bus/pci/devices/{device_id}/remove", "w") as f:
            f.write("1")
        # Add the device back
        with open("/sys/bus/pci/rescan", "w") as f:
            f.write("1")
    except Exception as e:
        print(f"Error hot-plugging device: {e}")
        return False
    return True

def test_hotplug():
    assert hotplug_nvidia(), "Failed to hot-plug device"

