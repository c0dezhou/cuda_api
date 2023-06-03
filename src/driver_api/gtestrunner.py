import os
import subprocess
import random
import re
from fuzzywuzzy import fuzz

class GTestRunner:
    def __init__(self):
        self.test_cases = []
    
    def compile_and_build(self):
        try:
            build_process = subprocess.run(['cmake', '.', '-B', 'build'], capture_output=True, text=True)
            if build_process.returncode != 0:
                print(f"Compilation failed with error:\n{build_process.stderr}")
                return

            compile_process = subprocess.run(['cmake', '--build', 'build'], capture_output=True, text=True)
            if compile_process.returncode != 0:
                print(f"Build failed with error:\n{compile_process.stderr}")
                return


            return exec_process.stdout
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None
        

    def get_test_files(self, dir_path='.'):
        test_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.cpp'):
                    test_files.append(os.path.join(root, file))
        return test_files

    def get_test_cases(self):
        test_files = self.get_test_files()
        for file in test_files:
            with open(file, 'r') as f:
                content = f.read()
                test_case_names = re.findall(r'TEST_F\((\w+),\s*(\w+)\)', content)
                for test_case in test_case_names:
                    self.test_cases.append(f"{test_case[0]}.{test_case[1]}")

    def runing_cases(self, test_filter=None):
        try:
            exec_process = subprocess.run(['./build/test', '--gtest_filter=' + test_filter],
                                          capture_output=True, text=True)
            if exec_process.returncode != 0:
                print(f"Execution failed with error:\n{exec_process.stderr}")
                return
            return exec_process.stdout
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    def run_tests_randomly(self):
        self.get_test_cases()
        random.shuffle(self.test_cases)
        for test_case in self.test_cases:
            print(f"Running test case {test_case}...")
            output = self.runing_cases(test_filter=test_case)
            if output:
                print("Output:\n", output)
                self.analyze_output(output)
    

    def run_tests_recursively(self, test_name_pattern):
        self.get_test_cases()
        matching_tests = [test_case for test_case in self.test_cases
                          if fuzz.partial_ratio(test_case, test_name_pattern) > 80]
        for test_case in matching_tests:
            print(f"Running test case {test_case}...")
            output = self.runing_cases(test_filter=test_case)
            if output:
                print("Output:\n", output)
                self.analyze_output(output)

    def run_stress_tests(self, num_tests):
        self.get_test_cases()
        for _ in range(num_tests):
            random.shuffle(self.test_cases)
            for test_case in self.test_cases:
                print(f"Running test case {test_case}...")
                output = self.runing_cases(test_filter=test_case)
                if output:
                    print("Output:\n", output)
                    self.analyze_output(output)

    def analyze_output(self, output):
        passed_tests = re.findall(r'\[ *OK *\] (.*) \(.* ms\)', output)
        skipped_tests = re.findall(r'\[ *SKIPPED *\] (.*) \(.* ms\)', output)
        failed_tests = re.findall(r'\[ *FAILED *\] (.*)', output)

        print(f"Passed Tests: {len(passed_tests)}")
        print(f"Skipped Tests: {len(skipped_tests)}")
        print(f"Failed Tests: {len(failed_tests)}")

        print("\nDetail:")
        if passed_tests:
            print("\nPassed:")
            for test in passed_tests:
                print(test)
        if skipped_tests:
            print("\nSkipped:")
            for test in skipped_tests:
                print(test)
        if failed_tests:
            print("\nFailed:")
            for test in failed_tests:
                print(test)

    def execute_file_and_check_return(self, file_path):
        try:
            result = subprocess.run([file_path], capture_output=True, text=True)
            print(f"Output:\n{result.stdout}")
            print(f"Errors (if any):\n{result.stderr}")
            print(f"Return code: {result.returncode}")
            if result.returncode != 0:
                print("The execution failed.")
            else:
                print("The execution was successful.")
        except Exception as e:
            print(f"An error occurred while executing the file: {e}")


    @staticmethod
    def handle_core_dump():
        # Set the core file size to unlimited so that a core dump will be produced upon crash
        subprocess.run(['ulimit', '-c', 'unlimited'])

        # Check for a core dump
        core_files = [f for f in os.listdir() if f.startswith('core')]
        for core_file in core_files:
            print(f"Core dump found: {core_file}")
            gdb_process = subprocess.run(['gdb', './build/test', core_file, '-ex', 'bt', '-batch'],
                                         capture_output=True, text=True)
            print(gdb_process.stdout)

    @staticmethod
    def check_dmesg_for_errors():
        dmesg_output = subprocess.run(['dmesg'], capture_output=True, text=True).stdout
        error_lines = re.findall(r'\[.*\] .* error .*', dmesg_output)

        if error_lines:
            print("\nError messages found in dmesg:")
            for line in error_lines:
                print(line)

    


if __name__ == "__main__":
    test_runner = GTestRunner()
    test_runner.compile_and_build()
    test_runner.run_tests_randomly()
    test_runner.run_tests_recursively("my_test_case")
    test_runner.run_stress_tests(10)
    test_runner.handle_core_dump()
    test_runner.check_dmesg_for_errors()
    test_runner.execute_file_and_check_return('./build/test')



