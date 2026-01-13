"""Task tracker for maintaining LLM awareness of multi-step tasks."""
import uuid
from typing import Optional


class TaskTracker:
    """In-memory task tracker for agent sessions.

    Maintains a list of tasks with status, allowing the LLM to track
    progress on multi-step plans without losing context.
    """

    VALID_STATUSES = {"pending", "in_progress", "completed"}

    def __init__(self):
        self.tasks = []  # List of {id, content, active_form, status}

    def add(self, content: str, active_form: str = None) -> str:
        """Add a new task. Returns the task ID."""
        task_id = str(uuid.uuid4())[:8]
        self.tasks.append({
            "id": task_id,
            "content": content,
            "active_form": active_form or content,
            "status": "pending"
        })
        return task_id

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
            lines.append(f"  {marker} {task['content']} ({task['id']})")

        return "\n".join(lines)

    def clear_completed(self):
        """Remove all completed tasks."""
        self.tasks = [t for t in self.tasks if t["status"] != "completed"]

    def clear_all(self):
        """Remove all tasks."""
        self.tasks = []


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
