"""Unit tests for TaskTracker."""
from task_tracker import TaskTracker, get_tracker, set_tracker, reset_tracker

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


def test_add_task(r: TestResults):
    """Adding a task returns an ID and task is pending."""
    tracker = TaskTracker()
    task_id = tracker.add("Read errol.py", "Reading errol.py")
    task = tracker.get(task_id)
    if task_id and task and task["status"] == "pending":
        r.ok("add_task")
    else:
        r.fail("add_task", f"Expected pending task, got: {task}")


def test_set_status_in_progress(r: TestResults):
    """Setting status to in_progress works."""
    tracker = TaskTracker()
    task_id = tracker.add("Read file", "Reading file")
    result = tracker.set_status(task_id, "in_progress")
    task = tracker.get(task_id)
    if result and task["status"] == "in_progress":
        r.ok("set_status_in_progress")
    else:
        r.fail("set_status_in_progress", f"Expected in_progress, got: {task}")


def test_set_status_completed(r: TestResults):
    """Setting status to completed works."""
    tracker = TaskTracker()
    task_id = tracker.add("Read file", "Reading file")
    result = tracker.set_status(task_id, "completed")
    task = tracker.get(task_id)
    if result and task["status"] == "completed":
        r.ok("set_status_completed")
    else:
        r.fail("set_status_completed", f"Expected completed, got: {task}")


def test_to_prompt_context_empty(r: TestResults):
    """Empty tracker returns empty context."""
    tracker = TaskTracker()
    context = tracker.to_prompt_context()
    if context == "":
        r.ok("to_prompt_context_empty")
    else:
        r.fail("to_prompt_context_empty", f"Expected empty string, got: {context!r}")


def test_to_prompt_context_formatting(r: TestResults):
    """Tasks are formatted correctly for prompt injection."""
    tracker = TaskTracker()
    id1 = tracker.add("Task one", "Doing task one")
    id2 = tracker.add("Task two", "Doing task two")
    tracker.set_status(id1, "completed")
    tracker.set_status(id2, "in_progress")
    context = tracker.to_prompt_context()
    has_completed = "[✓]" in context
    has_in_progress = "[→]" in context
    has_task_one = "Task one" in context
    has_task_two = "Task two" in context
    if has_completed and has_in_progress and has_task_one and has_task_two:
        r.ok("to_prompt_context_formatting")
    else:
        r.fail("to_prompt_context_formatting", f"Missing markers or tasks in: {context}")


def test_list_tasks(r: TestResults):
    """List returns all tasks."""
    tracker = TaskTracker()
    tracker.add("A", "a")
    tracker.add("B", "b")
    tasks = tracker.list()
    if len(tasks) == 2:
        r.ok("list_tasks")
    else:
        r.fail("list_tasks", f"Expected 2 tasks, got: {len(tasks)}")


def test_invalid_status_rejected(r: TestResults):
    """Invalid status values are rejected."""
    tracker = TaskTracker()
    task_id = tracker.add("Task", "Tasking")
    result = tracker.set_status(task_id, "invalid_status")
    task = tracker.get(task_id)
    if not result and task["status"] == "pending":
        r.ok("invalid_status_rejected")
    else:
        r.fail("invalid_status_rejected", f"Expected rejection, got result={result}, status={task['status']}")


def test_has_tasks(r: TestResults):
    """has_tasks returns correct boolean."""
    tracker = TaskTracker()
    if tracker.has_tasks():
        r.fail("has_tasks", "Expected False for empty tracker")
        return
    tracker.add("Task", "Tasking")
    if tracker.has_tasks():
        r.ok("has_tasks")
    else:
        r.fail("has_tasks", "Expected True after adding task")


def test_clear_completed(r: TestResults):
    """clear_completed removes only completed tasks."""
    tracker = TaskTracker()
    id1 = tracker.add("Task 1", "t1")
    id2 = tracker.add("Task 2", "t2")
    tracker.set_status(id1, "completed")
    tracker.clear_completed()
    tasks = tracker.list()
    if len(tasks) == 1 and tasks[0]["id"] == id2:
        r.ok("clear_completed")
    else:
        r.fail("clear_completed", f"Expected 1 task (id2), got: {tasks}")


def test_clear_all(r: TestResults):
    """clear_all removes all tasks."""
    tracker = TaskTracker()
    tracker.add("Task 1", "t1")
    tracker.add("Task 2", "t2")
    tracker.clear_all()
    if not tracker.has_tasks():
        r.ok("clear_all")
    else:
        r.fail("clear_all", f"Expected empty, got: {tracker.list()}")


def test_get_nonexistent(r: TestResults):
    """get returns None for nonexistent task."""
    tracker = TaskTracker()
    task = tracker.get("nonexistent_id")
    if task is None:
        r.ok("get_nonexistent")
    else:
        r.fail("get_nonexistent", f"Expected None, got: {task}")


def test_set_status_nonexistent(r: TestResults):
    """set_status returns False for nonexistent task."""
    tracker = TaskTracker()
    result = tracker.set_status("nonexistent_id", "completed")
    if not result:
        r.ok("set_status_nonexistent")
    else:
        r.fail("set_status_nonexistent", f"Expected False, got: {result}")


def test_global_tracker(r: TestResults):
    """Global tracker functions work correctly."""
    reset_tracker()
    tracker = get_tracker()
    task_id = tracker.add("Global test", "Testing global")
    tracker2 = get_tracker()
    if tracker2.get(task_id) is not None:
        r.ok("global_tracker")
    else:
        r.fail("global_tracker", "Global tracker not shared")


def run_all_tests() -> TestResults:
    """Run all tests and return results."""
    r = TestResults()

    test_add_task(r)
    test_set_status_in_progress(r)
    test_set_status_completed(r)
    test_to_prompt_context_empty(r)
    test_to_prompt_context_formatting(r)
    test_list_tasks(r)
    test_invalid_status_rejected(r)
    test_has_tasks(r)
    test_clear_completed(r)
    test_clear_all(r)
    test_get_nonexistent(r)
    test_set_status_nonexistent(r)
    test_global_tracker(r)

    return r


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\n{results.passed} passed, {results.failed} failed")
    if results.failed > 0:
        print("\nFailures:")
        for name, msg in results.errors:
            print(f"  {name}: {msg}")
        exit(1)
