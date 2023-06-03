import pytest
import subprocess

import pytest
import subprocess

def test_device_detection():
    lspci_output = subprocess.check_output(["lspci"]).decode()
    assert "NVIDIA Corporation" in lspci_output, "NVIDIA device not detected"

def test_device_details():
    lspci_output = subprocess.check_output(["lspci", "-v", "-s", "00:02.0"]).decode()
    assert "Kernel driver in use: nvidia" in lspci_output, "NVIDIA driver not in use for device"

def test_bus_speed_and_width():
    lspci_output = subprocess.check_output(["lspci", "-vv", "-s", "00:02.0"]).decode()
    assert "LnkSta: Speed" in lspci_output, "Link speed not found for device"
    assert "LnkSta: Width" in lspci_output, "Link width not found for device"


def test_device_identification():
    lspci_output = subprocess.check_output(["lspci", "-nn"]).decode()
    assert "[10de:" in lspci_output, "NVIDIA device not detected"

def test_driver_verification():
    lspci_output = subprocess.check_output(["lspci", "-k"]).decode()
    assert "Kernel driver in use: nvidia" in lspci_output, "NVIDIA driver not in use"

def test_device_configuration():
    lspci_output = subprocess.check_output(["lspci", "-vvv"]).decode()
    assert "Capabilities:" in lspci_output, "Capabilities of devices not found"

def test_bus_enumeration():
    lspci_output = subprocess.check_output(["lspci", "-t"]).decode()
    assert "[0000:00]" in lspci_output, "PCI bus hierarchy not found"

def test_device_hotplug_detection():
    lspci_output_before = subprocess.check_output(["lspci"]).decode()
    # Here, add the steps to perform hotplug of the device
    lspci_output_after = subprocess.check_output(["lspci"]).decode()
    assert lspci_output_before != lspci_output_after, "No changes detected in PCI devices after hotplug"

def test_system_validation():
    expected_devices = ["NVIDIA Corporation", "Intel Corporation"]
    lspci_output = subprocess.check_output(["lspci"]).decode()
    for device in expected_devices:
        assert device in lspci_output, f"{device} not found in system"

