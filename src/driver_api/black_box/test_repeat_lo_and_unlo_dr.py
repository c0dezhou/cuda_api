import pytest
import subprocess
import time

def is_nvidia_loaded():
    try:
        output = subprocess.check_output(["lsmod"], stderr=subprocess.STDOUT)
        return 'nvidia' in output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error checking module status: {e.output}")
        return False

def load_nvidia():
    if not is_nvidia_loaded():
        try:
            subprocess.check_output(["sudo", "modprobe", "nvidia"], stderr=subprocess.STDOUT)
            time.sleep(2)  # Give some time for the driver to load
        except subprocess.CalledProcessError as e:
            print(f"Error in loading nvidia: {e.output}")
            return False
    return is_nvidia_loaded()

def unload_nvidia():
    if is_nvidia_loaded():
        try:
            subprocess.check_output(["sudo", "modprobe", "-r", "nvidia"], stderr=subprocess.STDOUT)
            time.sleep(2)  # Give some time for the driver to unload
        except subprocess.CalledProcessError as e:
            print(f"Error in unloading nvidia: {e.output}")
            return False
    return not is_nvidia_loaded()

def test_nvidia_load_unload():
    for _ in range(5):  # Number of times to repeat the test
        assert load_nvidia() is True, "Failed to load the Nvidia driver"
        time.sleep(5)  # Let the driver initialize and settle
        assert unload_nvidia() is True, "Failed to unload the Nvidia driver"
        time.sleep(2)  # Give some time before loading the driver again

