import paramiko
import time
import pytest

HOST = 'your-target-machine'
PORT = 22
USERNAME = 'your-username'
PASSWORD = 'your-password'

def reboot_machine():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD)
    stdin, stdout, stderr = client.exec_command('sudo reboot')
    client.close()

def test_machine_restart():
    for i in range(5):  # replace with the number of times you want to restart
        reboot_machine()
        time.sleep(60)  # wait for the machine to restart, adjust this value based on your machine's startup time

    # After all reboots, you can perform some checks here, for example, check if Nvidia driver is up.
    # Connect to the machine again to run nvidia-smi or any other command
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD)
    stdin, stdout, stderr = client.exec_command('nvidia-smi')
    output = stdout.read().decode()
    client.close()

    assert "NVIDIA-SMI" in output, "Nvidia driver is not running correctly after machine restarts"
