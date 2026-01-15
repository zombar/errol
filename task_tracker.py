"""Task tracker for maintaining LLM awareness of multi-step tasks."""
import json
import uuid
from pathlib import Path
from typing import Optional


class TaskTracker:
    """Task tracker with persistence for agent sessions.

    Maintains a list of tasks with status, allowing the LLM to track
    progress on multi-step plans. Supports save/load for session recovery.
    """

    VALID_STATUSES = {"pending", "in_progress", "completed"}
    VALID_MODES = {"planning", "write", "validation"}

    def __init__(self):
        self.tasks = []  # List of {id, content, active_form, status, parent_id}
        self.original_task = ""  # The user's original request
        self.messages = []  # Conversation history for resumption
        self.interrupted = False  # Whether session was interrupted
        self.mode = "planning"  # planning, write, or validation
        self.plan_path = Path(".errol/PLAN.md")
        self.current_task_files = []  # Files modified during current task

    def add(self, content: str, active_form: str = None, parent_id: str = None,
            file_path: str = None, anchor: str = None) -> str:
        """Add a new task. If parent_id given, insert after parent and its subtasks.

        Args:
            content: Task description
            active_form: Display text (defaults to content)
            parent_id: Parent task ID for subtasks
            file_path: Optional file path this task affects
            anchor: Optional semantic anchor (function/class name) for locating edit
        """
        # Check for duplicate task with same content and parent
        for task in self.tasks:
            if task["content"] == content and task.get("parent_id") == parent_id:
                return task["id"]  # Return existing task ID instead of creating new

        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "content": content,
            "active_form": active_form or content,
            "status": "pending",
            "parent_id": parent_id,
            "file_path": file_path,
            "anchor": anchor
        }

        if parent_id:
            # Find insertion point: after parent and all its existing subtasks
            insert_idx = self._find_subtask_insert_point(parent_id)
            self.tasks.insert(insert_idx, task)
        else:
            self.tasks.append(task)

        return task_id

    def _find_subtask_insert_point(self, parent_id: str) -> int:
        """Find index to insert subtask: after parent and its existing subtasks."""
        parent_idx = None
        for i, task in enumerate(self.tasks):
            if task["id"] == parent_id:
                parent_idx = i
                break

        if parent_idx is None:
            return len(self.tasks)

        # Find last subtask of this parent
        insert_idx = parent_idx + 1
        while insert_idx < len(self.tasks):
            if self.tasks[insert_idx].get("parent_id") == parent_id:
                insert_idx += 1
            else:
                break

        return insert_idx

    def get(self, task_id: str) -> Optional[dict]:
        """Get a task by ID."""
        for task in self.tasks:
            if task["id"] == task_id:
                return task
        return None

    def set_status(self, task_id: str, status: str) -> bool:
        """Set task status. Returns True if successful."""
        if status not in self.VALID_STATUSES:
            return False
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = status
                return True
        return False

    def list(self) -> list[dict]:
        """Return all tasks."""
        return self.tasks.copy()

    def has_tasks(self) -> bool:
        """Check if there are any tasks."""
        return len(self.tasks) > 0

    def to_prompt_context(self) -> str:
        """Format tasks for inclusion in prompts.

        Returns empty string if no tasks, otherwise a formatted task list.
        """
        if not self.tasks:
            return ""

        lines = ["Current Tasks:"]
        for task in self.tasks:
            if task["status"] == "completed":
                marker = "[✓]"
            elif task["status"] == "in_progress":
                marker = "[→]"
            else:
                marker = "[ ]"

            # Build location hint if file_path or anchor present
            location = ""
            if task.get("file_path") or task.get("anchor"):
                parts = []
                if task.get("file_path"):
                    parts.append(task["file_path"])
                if task.get("anchor"):
                    parts.append(task["anchor"])
                location = f" @ {':'.join(parts)}"

            lines.append(f"  {marker} {task['content']}{location} ({task['id']})")

        return "\n".join(lines)

    def clear_completed(self):
        """Remove all completed tasks."""
        self.tasks = [t for t in self.tasks if t["status"] != "completed"]

    def clear_all(self):
        """Remove all tasks."""
        self.tasks = []

    def remove(self, task_id: str) -> bool:
        """Remove a task by ID. Returns True if removed."""
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                del self.tasks[i]
                return True
        return False

    def add_modified_file(self, file_path: str):
        """Track a file modified during current task."""
        if file_path and file_path not in self.current_task_files:
            self.current_task_files.append(file_path)

    def get_modified_files(self):
        """Get list of files modified during current task."""
        return self.current_task_files.copy()

    def clear_modified_files(self):
        """Clear the modified files list (after validation)."""
        self.current_task_files = []

    def save(self, path: Path = None):
        """Save state to disk. Also regenerates PLAN.md."""
        path = path or Path(".errol") / "last_session.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tasks": self.tasks,
            "original_task": self.original_task,
            "messages": self.messages,
            "interrupted": self.interrupted,
            "mode": self.mode
        }
        path.write_text(json.dumps(data, indent=2))

        # Regenerate PLAN.md from tasks
        if self.original_task:
            self.write_plan(self.original_task)

    def load(self, path: Path = None) -> bool:
        """Load state from disk. Returns True if loaded."""
        path = path or Path(".errol") / "last_session.json"
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            self.tasks = data.get("tasks", [])
            self.original_task = data.get("original_task", "")
            self.messages = data.get("messages", [])
            self.interrupted = data.get("interrupted", False)
            self.mode = data.get("mode", "planning")
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def write_plan(self, original_task: str, context: str = ""):
        """Write tasks to PLAN.md file."""
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"# Task: {original_task}", "", "## Plan"]

        # Number top-level tasks
        num = 1
        for task in self.tasks:
            if task["status"] == "completed":
                marker = "[x]"
            else:
                marker = "[ ]"

            # Build location hint if file_path or anchor present
            location = ""
            if task.get("file_path") or task.get("anchor"):
                parts = []
                if task.get("file_path"):
                    parts.append(f"`{task['file_path']}`")
                if task.get("anchor"):
                    parts.append(f"`{task['anchor']}`")
                location = f" ({' @ '.join(parts)})"

            if task.get("parent_id"):
                # Subtask - indented
                lines.append(f"   - {marker} {task['content']}{location}")
            else:
                # Top-level task
                lines.append(f"{num}. {marker} {task['content']}{location}")
                num += 1

        if context:
            lines.extend(["", "## Context", context])

        self.plan_path.write_text("\n".join(lines))


# Global instance for use with todo tool
_tracker_instance: Optional[TaskTracker] = None


def get_tracker() -> TaskTracker:
    """Get the global tracker instance, creating if needed."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = TaskTracker()
    return _tracker_instance


def set_tracker(tracker: TaskTracker):
    """Set the global tracker instance."""
    global _tracker_instance
    _tracker_instance = tracker


def reset_tracker():
    """Reset the global tracker to a new instance."""
    global _tracker_instance
    _tracker_instance = TaskTracker()
