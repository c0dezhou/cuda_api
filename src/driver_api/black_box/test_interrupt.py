import pytest
import subprocess

def read_interrupts():
    try:
        with open('/proc/interrupts', 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading /proc/interrupts: {e}")
        return None

def test_interrupts():
    interrupts_before = read_interrupts()
    assert interrupts_before is not None, "Failed to read /proc/interrupts"
    
    # Run your CUDA program here...
    subprocess.run(["/path/to/your/cuda/program"])
    
    interrupts_after = read_interrupts()
    assert interrupts_after is not None, "Failed to read /proc/interrupts"
    
    # Here you would add assertions to check that the interrupt counts have increased for your device.
    # This is device and setup specific, so it's hard to provide a generic example.

