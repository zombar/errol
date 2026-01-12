"""Unit tests for Errol tools - catches parameter handling issues."""
import tempfile
import os
from pathlib import Path

from tools import (
    read_file, write_file, edit_file, run_bash, glob_files,
    execute_tool
)
from errol import parse_tool_calls_from_text, extract_json_objects, looks_like_question


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  OK: {name}")

    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  FAIL: {name} - {msg}")


def test_read_file_missing_path(r: TestResults):
    """read_file should return error when path is missing."""
    result = read_file()
    if "Error" in result or "error" in result.lower():
        r.ok("read_file_missing_path")
    else:
        r.fail("read_file_missing_path", f"Expected error, got: {result}")


def test_read_file_null_string(r: TestResults):
    """read_file should handle 'null' string for offset/limit."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        path = f.name
    try:
        result = read_file(path=path, offset="null", limit="null")
        if "line1" in result and "Error" not in result:
            r.ok("read_file_null_string")
        else:
            r.fail("read_file_null_string", f"Expected content, got: {result[:100]}")
    finally:
        os.unlink(path)


def test_read_file_string_integers(r: TestResults):
    """read_file should handle string integers for offset/limit."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        path = f.name
    try:
        result = read_file(path=path, offset="0", limit="10")
        if "line1" in result and "Error" not in result:
            r.ok("read_file_string_integers")
        else:
            r.fail("read_file_string_integers", f"Expected content, got: {result[:100]}")
    finally:
        os.unlink(path)


def test_write_file_missing_content(r: TestResults):
    """write_file should return error when content is missing."""
    result = write_file(path="/tmp/test.txt")
    if "Error" in result and "content" in result.lower():
        r.ok("write_file_missing_content")
    else:
        r.fail("write_file_missing_content", f"Expected error about content, got: {result}")


def test_write_file_missing_path(r: TestResults):
    """write_file should return error when path is missing."""
    result = write_file(content="test")
    if "Error" in result and "path" in result.lower():
        r.ok("write_file_missing_path")
    else:
        r.fail("write_file_missing_path", f"Expected error about path, got: {result}")


def test_edit_file_missing_old_string(r: TestResults):
    """edit_file should return error when old_string is missing."""
    result = edit_file(path="/tmp/test.txt", new_string="new")
    if "Error" in result and "old_string" in result.lower():
        r.ok("edit_file_missing_old_string")
    else:
        r.fail("edit_file_missing_old_string", f"Expected error about old_string, got: {result}")


def test_edit_file_missing_new_string(r: TestResults):
    """edit_file should return error when new_string is missing."""
    result = edit_file(path="/tmp/test.txt", old_string="old")
    if "Error" in result and "new_string" in result.lower():
        r.ok("edit_file_missing_new_string")
    else:
        r.fail("edit_file_missing_new_string", f"Expected error about new_string, got: {result}")


def test_edit_file_identical_strings(r: TestResults):
    """edit_file should return error when old_string equals new_string."""
    result = edit_file(path="/tmp/test.txt", old_string="same", new_string="same")
    if "Error" in result and "identical" in result.lower():
        r.ok("edit_file_identical_strings")
    else:
        r.fail("edit_file_identical_strings", f"Expected error about identical, got: {result}")


def test_bash_missing_command(r: TestResults):
    """run_bash should return error when command is missing."""
    result = run_bash()
    if "Error" in result or "error" in result.lower():
        r.ok("bash_missing_command")
    else:
        r.fail("bash_missing_command", f"Expected error, got: {result}")


def test_bash_cmd_alias(r: TestResults):
    """run_bash should accept 'cmd' as alias for 'command'."""
    result = run_bash(cmd="echo hello")
    if "hello" in result:
        r.ok("bash_cmd_alias")
    else:
        r.fail("bash_cmd_alias", f"Expected 'hello', got: {result}")


def test_bash_string_timeout(r: TestResults):
    """run_bash should handle string timeout."""
    result = run_bash(command="echo test", timeout="10")
    if "test" in result:
        r.ok("bash_string_timeout")
    else:
        r.fail("bash_string_timeout", f"Expected 'test', got: {result}")


def test_bash_cmd_array(r: TestResults):
    """run_bash should handle cmd as array."""
    result = run_bash(cmd=["bash", "-c", "echo array_test"])
    if "array_test" in result:
        r.ok("bash_cmd_array")
    else:
        r.fail("bash_cmd_array", f"Expected 'array_test', got: {result}")


def test_execute_tool_filters_invalid_args(r: TestResults):
    """execute_tool should filter out invalid arguments."""
    # These extra args should be ignored, not cause a crash
    result = execute_tool("bash", {
        "command": "echo filtered",
        "function": "bash",
        "type": "function",
        "parameters": {},
        "invalid_arg": "should_be_ignored"
    })
    if "filtered" in result:
        r.ok("execute_tool_filters_invalid_args")
    else:
        r.fail("execute_tool_filters_invalid_args", f"Expected 'filtered', got: {result}")


def test_execute_tool_unknown_tool(r: TestResults):
    """execute_tool should return error for unknown tool."""
    result = execute_tool("nonexistent_tool", {})
    if "Error" in result and "Unknown tool" in result:
        r.ok("execute_tool_unknown_tool")
    else:
        r.fail("execute_tool_unknown_tool", f"Expected unknown tool error, got: {result}")


def test_extract_json_nested(r: TestResults):
    """extract_json_objects should handle nested braces."""
    text = 'Some text {"name": "test", "args": {"nested": "value"}} more text'
    objects = extract_json_objects(text)
    if len(objects) == 1 and '"nested"' in objects[0]:
        r.ok("extract_json_nested")
    else:
        r.fail("extract_json_nested", f"Expected nested JSON, got: {objects}")


def test_parse_tool_calls_parameters_key(r: TestResults):
    """parse_tool_calls_from_text should accept 'parameters' key."""
    text = '{"name": "bash", "parameters": {"command": "echo hi"}}'
    calls = parse_tool_calls_from_text(text)
    if len(calls) == 1 and calls[0]["function"]["name"] == "bash":
        r.ok("parse_tool_calls_parameters_key")
    else:
        r.fail("parse_tool_calls_parameters_key", f"Expected bash call, got: {calls}")


def test_parse_tool_calls_arguments_key(r: TestResults):
    """parse_tool_calls_from_text should accept 'arguments' key."""
    text = '{"name": "bash", "arguments": {"command": "echo hi"}}'
    calls = parse_tool_calls_from_text(text)
    if len(calls) == 1 and calls[0]["function"]["name"] == "bash":
        r.ok("parse_tool_calls_arguments_key")
    else:
        r.fail("parse_tool_calls_arguments_key", f"Expected bash call, got: {calls}")


def test_glob_files_basic(r: TestResults):
    """glob_files should find Python files."""
    result = glob_files("*.py")
    if "errol.py" in result or "tools.py" in result:
        r.ok("glob_files_basic")
    else:
        r.fail("glob_files_basic", f"Expected to find .py files, got: {result[:200]}")


def test_glob_files_missing_pattern(r: TestResults):
    """glob_files should return error when pattern is missing."""
    result = glob_files()
    if "Error" in result and "pattern" in result.lower():
        r.ok("glob_files_missing_pattern")
    else:
        r.fail("glob_files_missing_pattern", f"Expected error about pattern, got: {result}")


def test_looks_like_question_true(r: TestResults):
    """looks_like_question should detect questions requiring action."""
    test_cases = [
        "Which file should I start with?",
        "Do you want me to continue?",
        "Should I proceed with the changes?",
        "Please select an option.",
    ]
    all_passed = True
    for text in test_cases:
        if not looks_like_question(text):
            all_passed = False
            r.fail("looks_like_question_true", f"Should detect: {text}")
            break
    if all_passed:
        r.ok("looks_like_question_true")


def test_looks_like_question_false(r: TestResults):
    """looks_like_question should not detect completed responses."""
    test_cases = [
        "I have completed the task.",
        "Here are the files in the directory.",
        "The file has been created successfully.",
        "Let me know if you need anything else.",  # Polite closing, not a question
        "",
    ]
    all_passed = True
    for text in test_cases:
        if looks_like_question(text):
            all_passed = False
            r.fail("looks_like_question_false", f"Should not detect: {text}")
            break
    if all_passed:
        r.ok("looks_like_question_false")


def run_all_tests() -> TestResults:
    """Run all tests and return results."""
    r = TestResults()

    print("\nRunning tool tests...")

    # Parameter validation tests
    test_read_file_missing_path(r)
    test_read_file_null_string(r)
    test_read_file_string_integers(r)
    test_write_file_missing_content(r)
    test_write_file_missing_path(r)
    test_edit_file_missing_old_string(r)
    test_edit_file_missing_new_string(r)
    test_edit_file_identical_strings(r)
    test_bash_missing_command(r)
    test_bash_cmd_alias(r)
    test_bash_string_timeout(r)
    test_bash_cmd_array(r)

    # execute_tool tests
    test_execute_tool_filters_invalid_args(r)
    test_execute_tool_unknown_tool(r)

    # JSON parsing tests
    test_extract_json_nested(r)
    test_parse_tool_calls_parameters_key(r)
    test_parse_tool_calls_arguments_key(r)

    # Basic functionality tests
    test_glob_files_basic(r)
    test_glob_files_missing_pattern(r)

    # Question detection tests
    test_looks_like_question_true(r)
    test_looks_like_question_false(r)

    return r


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\n{results.passed} passed, {results.failed} failed")
    if results.failed > 0:
        print("\nFailures:")
        for name, msg in results.errors:
            print(f"  {name}: {msg}")
        exit(1)
