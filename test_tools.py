"""Unit tests for Errol tools - catches parameter handling issues."""
import tempfile
import os
from pathlib import Path

from tools import (
    read_file, write_file, edit_file, run_bash, glob_files,
    execute_tool, resolve_tool_name, TOOL_ALIASES
)
from errol import (
    parse_tool_calls_from_text, extract_json_objects, looks_like_question,
    looks_like_multi_step_plan, DIM, RESET, GREEN, RED, YELLOW
)


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  {GREEN}✓{RESET} {DIM}{name}{RESET}")

    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  {RED}✗{RESET} {name} {DIM}- {msg}{RESET}")


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
    """edit_file should return error when old_string is missing (with real file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world\n")
        tmp_path = f.name
    try:
        result = edit_file(path=tmp_path, new_string="new")
        if "Error" in result and "old_string" in result.lower():
            r.ok("edit_file_missing_old_string")
        else:
            r.fail("edit_file_missing_old_string", f"Expected error about old_string, got: {result}")
    finally:
        import os
        os.unlink(tmp_path)


def test_edit_file_missing_new_string(r: TestResults):
    """edit_file should return error when new_string is missing (with real file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world\n")
        tmp_path = f.name
    try:
        result = edit_file(path=tmp_path, old_string="hello")
        if "Error" in result and "new_string" in result.lower():
            r.ok("edit_file_missing_new_string")
        else:
            r.fail("edit_file_missing_new_string", f"Expected error about new_string, got: {result}")
    finally:
        import os
        os.unlink(tmp_path)


def test_edit_file_identical_strings(r: TestResults):
    """edit_file should return error when old_string equals new_string (with real file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("same content here\n")
        tmp_path = f.name
    try:
        result = edit_file(path=tmp_path, old_string="same", new_string="same")
        if "Error" in result and "identical" in result.lower():
            r.ok("edit_file_identical_strings")
        else:
            r.fail("edit_file_identical_strings", f"Expected error about identical, got: {result}")
    finally:
        import os
        os.unlink(tmp_path)


def test_edit_file_line_based(r: TestResults):
    """edit_file should support line_start/line_end selection."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("line 1\nline 2\nline 3\n")
        tmp_path = f.name
    try:
        result = edit_file(path=tmp_path, line_start=2, line_end=2, new_string="replaced line 2\n")
        if "Edited" in result or "applied" in result.lower():
            # Verify the content
            with open(tmp_path) as f:
                content = f.read()
            if "replaced line 2" in content and "line 1" in content and "line 3" in content:
                r.ok("edit_file_line_based")
            else:
                r.fail("edit_file_line_based", f"Content not as expected: {content}")
        else:
            r.fail("edit_file_line_based", f"Expected success, got: {result}")
    finally:
        import os
        os.unlink(tmp_path)


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
    # Use absolute path to this test file's directory
    test_dir = Path(__file__).parent.resolve()
    result = glob_files("*.py", str(test_dir))
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


def test_edit_file_tabs_to_spaces(r: TestResults):
    """edit_file should match file with tabs when search uses spaces."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("\tdef foo():\n\t\tpass\n")  # tabs
        path = f.name
    try:
        result = edit_file(
            path=path,
            old_string="    def foo():\n        pass",  # spaces (4 and 8)
            new_string="    def bar():\n        pass"
        )
        content = Path(path).read_text()
        if "def bar" in content:
            r.ok("edit_file_tabs_to_spaces")
        else:
            r.fail("edit_file_tabs_to_spaces", f"Expected 'def bar', got: {content!r}")
    finally:
        os.unlink(path)


def test_edit_file_spaces_to_tabs(r: TestResults):
    """edit_file should match file with spaces when search uses tabs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("    def foo():\n        return 1\n")  # spaces
        path = f.name
    try:
        result = edit_file(
            path=path,
            old_string="\tdef foo():\n\t\treturn 1",  # tabs
            new_string="\tdef bar():\n\t\treturn 2"
        )
        content = Path(path).read_text()
        if "def bar" in content and "return 2" in content:
            r.ok("edit_file_spaces_to_tabs")
        else:
            r.fail("edit_file_spaces_to_tabs", f"Expected 'def bar', got: {content!r}")
    finally:
        os.unlink(path)


def test_edit_file_preserves_indent_style(r: TestResults):
    """edit_file should preserve the file's original indentation style."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("\tdef foo():\n\t\treturn 1\n")  # tabs
        path = f.name
    try:
        result = edit_file(
            path=path,
            old_string="    def foo():\n        return 1",  # spaces
            new_string="    def bar():\n        return 2"   # spaces
        )
        content = Path(path).read_text()
        # Should preserve tabs from original file
        if "\tdef bar" in content and "\t\treturn 2" in content:
            r.ok("edit_file_preserves_indent_style")
        else:
            r.fail("edit_file_preserves_indent_style", f"Expected tabs preserved, got: {content!r}")
    finally:
        os.unlink(path)


def test_validate_uses_flexible_fallback(r: TestResults):
    """validate_tool_call should pass when flexible matching would succeed."""
    from tools import validate_tool_call
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("\tdef foo():\n\t\tpass\n")  # tabs
        path = f.name
    try:
        result = validate_tool_call("edit_file", {
            "path": path,
            "old_string": "    def foo():\n        pass",  # spaces
            "new_string": "    def bar():\n        pass"
        })
        if result is None:
            r.ok("validate_uses_flexible_fallback")
        else:
            r.fail("validate_uses_flexible_fallback", f"Expected None (valid), got: {result}")
    finally:
        os.unlink(path)


def test_resolve_tool_name_exact(r: TestResults):
    """resolve_tool_name should return exact match."""
    name, was_fuzzy = resolve_tool_name("bash")
    if name == "bash" and not was_fuzzy:
        r.ok("resolve_tool_name_exact")
    else:
        r.fail("resolve_tool_name_exact", f"Expected ('bash', False), got: ({name}, {was_fuzzy})")


def test_resolve_tool_name_alias(r: TestResults):
    """resolve_tool_name should resolve aliases."""
    name, was_fuzzy = resolve_tool_name("search")
    if name == "grep" and not was_fuzzy:
        r.ok("resolve_tool_name_alias")
    else:
        r.fail("resolve_tool_name_alias", f"Expected ('grep', False), got: ({name}, {was_fuzzy})")


def test_resolve_tool_name_fuzzy(r: TestResults):
    """resolve_tool_name should fuzzy match typos."""
    name, was_fuzzy = resolve_tool_name("greb")  # typo for grep
    if name == "grep" and was_fuzzy:
        r.ok("resolve_tool_name_fuzzy")
    else:
        r.fail("resolve_tool_name_fuzzy", f"Expected ('grep', True), got: ({name}, {was_fuzzy})")


def test_resolve_tool_name_unknown(r: TestResults):
    """resolve_tool_name should return original for unknown tools."""
    name, was_fuzzy = resolve_tool_name("totally_unknown_xyz")
    if name == "totally_unknown_xyz" and not was_fuzzy:
        r.ok("resolve_tool_name_unknown")
    else:
        r.fail("resolve_tool_name_unknown", f"Expected ('totally_unknown_xyz', False), got: ({name}, {was_fuzzy})")


def test_execute_tool_with_alias(r: TestResults):
    """execute_tool should work with tool aliases."""
    result = execute_tool("run", {"command": "echo hello_alias_test"})
    if "hello_alias_test" in result:
        r.ok("execute_tool_with_alias")
    else:
        r.fail("execute_tool_with_alias", f"Expected 'hello_alias_test', got: {result}")


def test_tool_aliases_defined(r: TestResults):
    """TOOL_ALIASES should have common aliases defined."""
    expected = ["search", "find", "cat", "run", "shell"]
    missing = [a for a in expected if a not in TOOL_ALIASES]
    if not missing:
        r.ok("tool_aliases_defined")
    else:
        r.fail("tool_aliases_defined", f"Missing aliases: {missing}")


def test_looks_like_multi_step_plan_numbered(r: TestResults):
    """looks_like_multi_step_plan should detect numbered steps."""
    text = """Here's the plan:
1. Create the Impact class
2. Add the impacts list
3. Spawn impacts on collision
4. Draw impacts each frame"""
    if looks_like_multi_step_plan(text):
        r.ok("looks_like_multi_step_plan_numbered")
    else:
        r.fail("looks_like_multi_step_plan_numbered", "Expected True for numbered steps")


def test_looks_like_multi_step_plan_single(r: TestResults):
    """looks_like_multi_step_plan should return False for single change."""
    text = "Change the color from WHITE to RED in the draw function."
    if not looks_like_multi_step_plan(text):
        r.ok("looks_like_multi_step_plan_single")
    else:
        r.fail("looks_like_multi_step_plan_single", "Expected False for single change")


def run_all_tests() -> TestResults:
    """Run all tests and return results."""
    r = TestResults()

    # Parameter validation tests
    test_read_file_missing_path(r)
    test_read_file_null_string(r)
    test_read_file_string_integers(r)
    test_write_file_missing_content(r)
    test_write_file_missing_path(r)
    test_edit_file_missing_old_string(r)
    test_edit_file_missing_new_string(r)
    test_edit_file_identical_strings(r)
    test_edit_file_line_based(r)
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

    # Multi-step plan detection tests
    test_looks_like_multi_step_plan_numbered(r)
    test_looks_like_multi_step_plan_single(r)

    # Indentation handling tests
    test_edit_file_tabs_to_spaces(r)
    test_edit_file_spaces_to_tabs(r)
    test_edit_file_preserves_indent_style(r)
    test_validate_uses_flexible_fallback(r)

    # Tool name resolution tests
    test_resolve_tool_name_exact(r)
    test_resolve_tool_name_alias(r)
    test_resolve_tool_name_fuzzy(r)
    test_resolve_tool_name_unknown(r)
    test_execute_tool_with_alias(r)
    test_tool_aliases_defined(r)

    return r


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\n{results.passed} passed, {results.failed} failed")
    if results.failed > 0:
        print("\nFailures:")
        for name, msg in results.errors:
            print(f"  {name}: {msg}")
        exit(1)
