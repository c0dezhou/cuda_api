import os
import subprocess
import random
import re
import psutil
import time

class SystemMonitor:
    def __init__(self):
        self.gpu_processes = {}
        self.cpu_percent = 0.0
        self.monitoring_data = []

    def start_gpu_monitor(self):
        self.gpu_processes = {}
        try:
            gpu_info_process = subprocess.Popen(['nvidia-smi', '--query-gpu=pid,utilization.gpu,utilization.memory,power.draw',
                                                 '--format=csv,noheader,nounits'], stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, universal_newlines=True)

            for line in iter(gpu_info_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    pid, gpu_util, mem_util, power_draw = line.split(',')
                    self.gpu_processes[int(pid)] = {'GPU Utilization (%)': float(gpu_util),
                                                    'Memory Utilization (%)': float(mem_util),
                                                    'Power Consumption (W)': float(power_draw)}

            gpu_info_process.stdout.close()
            gpu_info_process.wait()
        except Exception as e:
            print(f"Exception occurred while monitoring GPU: {e}")

    def get_cpu_percent(self):
        self.cpu_percent = psutil.cpu_percent()

    def monitor_system(self, duration):
        self.start_gpu_monitor()
        self.get_cpu_percent()

        print("\nSystem Monitoring:")
        print("-" * 100)
        print("Time\t\t\tCPU Utilization (%)\tGPU Utilization (%)\tMemory Utilization (%)\tPower Consumption (W)")
        print("-" * 100)

        start_time = time.time()
        while time.time() - start_time < duration:
            current_time = time.strftime("%H:%M:%S", time.localtime())

            cpu_info = f"{current_time}\t\t{self.cpu_percent:.2f}\t\t\t\t\t\t"
            gpu_info = ""

            for pid, gpu_stats in self.gpu_processes.items():
                gpu_util = gpu_stats['GPU Utilization (%)']
                mem_util = gpu_stats['Memory Utilization (%)']
                power_draw = gpu_stats['Power Consumption (W)']
                gpu_info += f"{pid}: {gpu_util:.2f} / {mem_util:.2f} / {power_draw:.2f}\t\t"

            print(cpu_info + gpu_info)
            self.monitoring_data.append((current_time, self.cpu_percent, self.gpu_processes.copy()))
            time.sleep(1)
            self.get_cpu_percent()
            self.start_gpu_monitor()

    def print_monitoring_data(self):
        print("\nFinal System Monitoring Data:")
        print("-" * 100)
        print("Time\t\t\tCPU Utilization (%)\tGPU Utilization (%)\tMemory Utilization (%)\tPower Consumption (W)")
        print("-" * 100)

        for data in self.monitoring_data:
            current_time, cpu_percent, gpu_processes = data

            cpu_info = f"{current_time}\t\t{cpu_percent:.2f}\t\t\t\t\t\t"
            gpu_info = ""

            for pid, gpu_stats in gpu_processes.items():
                gpu_util = gpu_stats['GPU Utilization (%)']
                mem_util = gpu_stats['Memory Utilization (%)']
                power_draw = gpu_stats['Power Consumption (W)']
                gpu_info += f"{pid}: {gpu_util:.2f} / {mem_util:.2f} / {power_draw:.2f}\t\t"

            print(cpu_info + gpu_info)

if __name__ == "__main__":
    test_runner = GTestRunner()
    test_runner.run_tests_randomly()
    test_runner.handle_core_dump('your_test_file.cpp')  # Replace with your test file name
    test_runner.check_dmesg_for_errors()

    # Example usage of running tests recursively with fuzzy matching
    test_runner.run_tests_recursively("my_test_case")

    # Example usage of running stress tests
    test_runner.run_stress_tests(10)  # Run 10 stress tests

    # Monitor system during test execution
    monitor = SystemMonitor()
    monitor.monitor_system(duration=10)  # Monitor for 10 seconds
    monitor.print_monitoring_data()  # Print the final monitoring data
