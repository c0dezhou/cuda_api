# 可以使用“nvidia-smi”命令在挂起/恢复周期之前和之后监控 NVIDIA 驱动程序的状态。 `nvidia-smi` 命令提供有关 GPU 的利用率、温度、功耗和其他统计数据的信息。
import pytest
import subprocess
import time

def get_gpu_stats():
    try:
        # Run nvidia-smi command and get its output
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]).decode()
        utilization, temperature = output.strip().split(', ')
        return int(utilization), int(temperature)
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None, None

def suspend_and_resume():
    try:
        # Suspend the system
        subprocess.check_call(["sudo", "systemctl", "suspend"])
        
        # Wait for a few seconds...
        time.sleep(10)
        
        # The system should automatically resume after being suspended.
        # If it doesn't, you may need to manually wake it up, for example by pressing a key or moving the mouse.
        
    except subprocess.CalledProcessError as e:
        print(f"Error suspending/resuming the system: {e}")
        return False
    return True

def test_suspend_and_resume():
    utilization_before, temperature_before = get_gpu_stats()
    assert utilization_before is not None, "Failed to get GPU utilization before suspend"
    assert temperature_before is not None, "Failed to get GPU temperature before suspend"
    
    assert suspend_and_resume(), "Failed to suspend and resume the system"
    
    utilization_after, temperature_after = get_gpu_stats()
    assert utilization_after is not None, "Failed to get GPU utilization after resume"
    assert temperature_after is not None, "Failed to get GPU temperature after resume"
    
    # Verify that the GPU utilization and temperature are within expected ranges
    assert 0 <= utilization_before <= 100, "GPU utilization before suspend is out of range"
    assert 0 <= temperature_before <= 100, "GPU temperature before suspend is out of range"
    assert 0 <= utilization_after <= 100, "GPU utilization after resume is out of range"
    assert 0 <= temperature_after <= 100, "GPU temperature after resume is out of range"

