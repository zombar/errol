"""Unit tests for validation stage."""
import io
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from task_tracker import TaskTracker


@contextmanager
def suppress_output():
    """Suppress stdout during tests."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# ANSI colors
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"


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


# --- File Tracking Tests ---

def test_add_modified_file(r: TestResults):
    """add_modified_file tracks file paths."""
    tracker = TaskTracker()
    tracker.add_modified_file("test.py")
    files = tracker.get_modified_files()
    if files == ["test.py"]:
        r.ok("add_modified_file")
    else:
        r.fail("add_modified_file", f"Expected ['test.py'], got: {files}")


def test_add_modified_file_no_duplicates(r: TestResults):
    """add_modified_file does not add duplicates."""
    tracker = TaskTracker()
    tracker.add_modified_file("test.py")
    tracker.add_modified_file("test.py")
    files = tracker.get_modified_files()
    if files == ["test.py"]:
        r.ok("add_modified_file_no_duplicates")
    else:
        r.fail("add_modified_file_no_duplicates", f"Expected ['test.py'], got: {files}")


def test_add_modified_file_multiple(r: TestResults):
    """add_modified_file tracks multiple files."""
    tracker = TaskTracker()
    tracker.add_modified_file("a.py")
    tracker.add_modified_file("b.py")
    files = tracker.get_modified_files()
    if files == ["a.py", "b.py"]:
        r.ok("add_modified_file_multiple")
    else:
        r.fail("add_modified_file_multiple", f"Expected ['a.py', 'b.py'], got: {files}")


def test_add_modified_file_ignores_empty(r: TestResults):
    """add_modified_file ignores empty strings."""
    tracker = TaskTracker()
    tracker.add_modified_file("")
    tracker.add_modified_file(None)
    files = tracker.get_modified_files()
    if files == []:
        r.ok("add_modified_file_ignores_empty")
    else:
        r.fail("add_modified_file_ignores_empty", f"Expected [], got: {files}")


def test_get_modified_files_returns_copy(r: TestResults):
    """get_modified_files returns a copy, not the original list."""
    tracker = TaskTracker()
    tracker.add_modified_file("test.py")
    files = tracker.get_modified_files()
    files.append("other.py")
    original = tracker.get_modified_files()
    if original == ["test.py"]:
        r.ok("get_modified_files_returns_copy")
    else:
        r.fail("get_modified_files_returns_copy", f"Expected ['test.py'], got: {original}")


def test_clear_modified_files(r: TestResults):
    """clear_modified_files clears the list."""
    tracker = TaskTracker()
    tracker.add_modified_file("test.py")
    tracker.clear_modified_files()
    files = tracker.get_modified_files()
    if files == []:
        r.ok("clear_modified_files")
    else:
        r.fail("clear_modified_files", f"Expected [], got: {files}")


# --- Validation Mode Tests ---

def test_validation_mode_valid(r: TestResults):
    """validation is a valid mode."""
    tracker = TaskTracker()
    if "validation" in TaskTracker.VALID_MODES:
        r.ok("validation_mode_valid")
    else:
        r.fail("validation_mode_valid", f"'validation' not in VALID_MODES: {TaskTracker.VALID_MODES}")


def test_set_validation_mode(r: TestResults):
    """Can set mode to validation."""
    tracker = TaskTracker()
    tracker.mode = "validation"
    if tracker.mode == "validation":
        r.ok("set_validation_mode")
    else:
        r.fail("set_validation_mode", f"Expected 'validation', got: {tracker.mode}")


# --- run_validation_stage Tests ---

def test_run_validation_stage_no_files(r: TestResults):
    """run_validation_stage returns True when no modified files."""
    from errol import run_validation_stage
    tracker = TaskTracker()
    config = {"ollama": {"host": "http://localhost:11434", "timeout": 300},
              "models": {"writing": "test-model"},
              "agent": {"validation": {"enabled": True, "model": None}}}
    with suppress_output():
        result = run_validation_stage(
            {"content": "Test task"},
            [],  # No modified files
            config,
            tracker,
            []
        )
    if result is True:
        r.ok("run_validation_stage_no_files")
    else:
        r.fail("run_validation_stage_no_files", f"Expected True, got: {result}")


def test_run_validation_stage_no_issues(r: TestResults):
    """run_validation_stage returns True when LLM reports no issues."""
    from errol import run_validation_stage
    tracker = TaskTracker()
    config = {"ollama": {"host": "http://localhost:11434", "timeout": 300},
              "models": {"writing": "test-model"},
              "agent": {"validation": {"enabled": True, "model": None}}}

    # Mock the LLM client
    with patch('errol.OllamaClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat_sync.return_value = {
            "message": {
                "content": "NO ISSUES FOUND",
                "tool_calls": []
            }
        }

        with suppress_output():
            result = run_validation_stage(
                {"content": "Test task"},
                ["test.py"],
                config,
                tracker,
                []
            )

        if result is True:
            r.ok("run_validation_stage_no_issues")
        else:
            r.fail("run_validation_stage_no_issues", f"Expected True, got: {result}")


def test_run_validation_stage_issues_skip(r: TestResults):
    """run_validation_stage returns True when user skips issues."""
    from errol import run_validation_stage
    tracker = TaskTracker()
    config = {"ollama": {"host": "http://localhost:11434", "timeout": 300},
              "models": {"writing": "test-model"},
              "agent": {"validation": {"enabled": True, "model": None}}}

    with patch('errol.OllamaClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat_sync.return_value = {
            "message": {
                "content": "ISSUES FOUND: syntax error on line 5",
                "tool_calls": []
            }
        }

        # Mock user input to skip
        with patch('builtins.input', return_value='n'):
            with suppress_output():
                result = run_validation_stage(
                    {"content": "Test task"},
                    ["test.py"],
                    config,
                    tracker,
                    []
                )

            if result is True:
                r.ok("run_validation_stage_issues_skip")
            else:
                r.fail("run_validation_stage_issues_skip", f"Expected True (skipped), got: {result}")


def test_run_validation_stage_issues_fix(r: TestResults):
    """run_validation_stage returns False when user wants to fix."""
    from errol import run_validation_stage
    tracker = TaskTracker()
    config = {"ollama": {"host": "http://localhost:11434", "timeout": 300},
              "models": {"writing": "test-model"},
              "agent": {"validation": {"enabled": True, "model": None}}}
    messages = []

    with patch('errol.OllamaClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat_sync.return_value = {
            "message": {
                "content": "ISSUES FOUND: syntax error on line 5",
                "tool_calls": []
            }
        }

        # Mock user input to fix
        with patch('builtins.input', return_value='y'):
            with suppress_output():
                result = run_validation_stage(
                    {"content": "Test task"},
                    ["test.py"],
                    config,
                    tracker,
                    messages
                )

            # Should return False and add fix request to messages
            if result is False and len(messages) == 1 and "VALIDATION ISSUES" in messages[0]["content"]:
                r.ok("run_validation_stage_issues_fix")
            else:
                r.fail("run_validation_stage_issues_fix", f"Expected False with fix message, got: result={result}, messages={messages}")


def run_all_tests() -> TestResults:
    r = TestResults()
    print("Running validation tests...")

    # File tracking tests
    test_add_modified_file(r)
    test_add_modified_file_no_duplicates(r)
    test_add_modified_file_multiple(r)
    test_add_modified_file_ignores_empty(r)
    test_get_modified_files_returns_copy(r)
    test_clear_modified_files(r)

    # Mode tests
    test_validation_mode_valid(r)
    test_set_validation_mode(r)

    # run_validation_stage tests
    test_run_validation_stage_no_files(r)
    test_run_validation_stage_no_issues(r)
    test_run_validation_stage_issues_skip(r)
    test_run_validation_stage_issues_fix(r)

    return r


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\n{results.passed} passed, {results.failed} failed")
    if results.failed > 0:
        print("\nFailures:")
        for name, msg in results.errors:
            print(f"  {name}: {msg}")
        exit(1)
