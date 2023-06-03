import pytest
import os

def run_test_program():
    # Path to the test program
    test_program = "/path/to/test_program"

    # Fork a new process
    pid = os.fork()

    if pid > 0:
        # In the parent process: wait for the child process to finish
        os.wait()
    else:
        # In the child process: run the test program
        os.execl(test_program, test_program)
        # If execl returns, there was an error
        print(f"Error running {test_program}")
        os._exit(1)

def test_run_test_program():
    try:
        run_test_program()
    except Exception as e:
        pytest.fail(f"Failed to run test program: {e}")

