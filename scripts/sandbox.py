#!/usr/bin/env python3
"""Safe code execution sandbox for HumanEval evaluation."""

import multiprocessing
import signal
import sys
from typing import Tuple, Optional


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Execution timed out")


def execute_code_in_process(code: str, test_code: str, timeout: int = 3) -> Tuple[bool, str]:
    """
    Execute code in a separate process with timeout.

    Args:
        code: The generated code to execute
        test_code: Test cases to run
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    def run_test():
        try:
            # Create isolated namespace
            namespace = {}

            # Execute the generated code
            exec(code, namespace)

            # Execute test code
            exec(test_code, namespace)

            return True, ""
        except Exception as e:
            return False, str(e)

    # Run in separate process with timeout
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    def worker():
        success, error = run_test()
        return_dict['success'] = success
        return_dict['error'] = error

    process = multiprocessing.Process(target=worker)
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        # Timeout occurred
        process.terminate()
        process.join()
        return False, f"Timeout after {timeout} seconds"

    if process.exitcode != 0:
        return False, f"Process exited with code {process.exitcode}"

    return return_dict.get('success', False), return_dict.get('error', 'Unknown error')


def check_correctness(
    code: str,
    test_code: str,
    timeout: int = 3,
) -> dict:
    """
    Check if generated code passes the test cases.

    Args:
        code: Generated code to test
        test_code: Test assertions
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with test results
    """
    try:
        passed, error = execute_code_in_process(code, test_code, timeout)

        return {
            "passed": passed,
            "error": error if not passed else None,
        }
    except Exception as e:
        return {
            "passed": False,
            "error": f"Sandbox error: {str(e)}",
        }


def run_single_test(code: str, test_input: str, expected_output: str, timeout: int = 3) -> bool:
    """
    Run a single test case.

    Args:
        code: Generated code
        test_input: Input for the test
        expected_output: Expected result
        timeout: Timeout in seconds

    Returns:
        True if test passes, False otherwise
    """
    test_code = f"""
    result = {test_input}
    assert result == {expected_output}, f"Expected {expected_output}, got {{result}}"
    """

    result = check_correctness(code, test_code, timeout)
    return result["passed"]
