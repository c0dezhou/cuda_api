import pytest
import os

def run_test_program():
    # Path to the test program
    test_program = "/path/to/test_program"

    # Spawn a new process and run the test program in it
    pid = os.spawnl(os.P_NOWAIT, test_program, test_program)

    if pid == 0:
        pytest.fail("Failed to spawn test program")

    # Wait for the spawned process to finish
    _, exit_status = os.waitpid(pid, 0)

    # Check the exit status
    if os.WIFEXITED(exit_status) and os.WEXITSTATUS(exit_status) == 0:
        # The process exited normally with a status of 0, which usually indicates success
        return
    else:
        pytest.fail("Test program exited with an error")

def test_run_test_program():
    try:
        run_test_program()
    except Exception as e:
        pytest.fail(f"Failed to run test program: {e}")

