"""Todo manager with JSON persistence for MoE orchestration."""
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List

class TodoManager:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path.home() / ".errol_todos.json"
        self.items: dict[str, dict] = {}
        self.load()

    def add(self, content: str, tier: str = "medium", parent: str = None,
            dependencies: List[str] = None, context: str = "") -> str:
        """Add a todo, return its ID."""
        id = uuid.uuid4().hex[:8]
        self.items[id] = {
            "id": id,
            "content": content,
            "status": "pending",
            "tier": tier,  # small|medium|large
            "parent": parent,  # parent task ID
            "dependencies": dependencies or [],  # task IDs that must complete first
            "context": context,  # input context for worker
            "result": "",  # output from worker
            "artifacts": [],  # files created/modified
            "created": datetime.now().isoformat(),
            "completed": None
        }
        self.save()
        return id

    def list(self, status: Optional[str] = None) -> List[dict]:
        """List todos, optionally filtered by status."""
        items = list(self.items.values())
        if status:
            items = [i for i in items if i["status"] == status]
        return sorted(items, key=lambda x: x["created"])

    def get(self, id: str) -> Optional[dict]:
        """Get a todo by ID."""
        return self.items.get(id)

    def complete(self, id: str) -> bool:
        """Mark a todo as complete."""
        if id in self.items:
            self.items[id]["status"] = "complete"
            self.items[id]["completed"] = datetime.now().isoformat()
            self.save()
            return True
        return False

    def start(self, id: str) -> bool:
        """Mark a todo as in progress."""
        if id in self.items:
            self.items[id]["status"] = "in_progress"
            self.save()
            return True
        return False

    def remove(self, id: str) -> bool:
        """Remove a todo."""
        if id in self.items:
            del self.items[id]
            self.save()
            return True
        return False

    def clear_completed(self) -> int:
        """Remove all completed todos."""
        before = len(self.items)
        self.items = {k: v for k, v in self.items.items() if v["status"] != "complete"}
        self.save()
        return before - len(self.items)

    def clear_all(self) -> int:
        """Remove all todos."""
        count = len(self.items)
        self.items = {}
        self.save()
        return count

    def update(self, id: str, **kwargs) -> bool:
        """Update todo fields."""
        if id not in self.items:
            return False
        for key, value in kwargs.items():
            if key in self.items[id]:
                self.items[id][key] = value
        self.save()
        return True

    def add_dependency(self, id: str, dep_id: str) -> bool:
        """Add a dependency to a task."""
        if id not in self.items or dep_id not in self.items:
            return False
        if dep_id not in self.items[id]["dependencies"]:
            self.items[id]["dependencies"].append(dep_id)
            self.save()
        return True

    def get_ready_tasks(self) -> List[dict]:
        """Get pending tasks whose dependencies are all complete."""
        ready = []
        for item in self.items.values():
            if item["status"] != "pending":
                continue
            # Check all dependencies are complete
            deps_complete = all(
                self.items.get(dep_id, {}).get("status") == "complete"
                for dep_id in item.get("dependencies", [])
            )
            if deps_complete:
                ready.append(item)
        return sorted(ready, key=lambda x: x["created"])

    def get_next_task(self) -> Optional[dict]:
        """Get the next ready task to execute."""
        ready = self.get_ready_tasks()
        return ready[0] if ready else None

    def set_result(self, id: str, result: str, artifacts: List[str] = None) -> bool:
        """Store worker result for a task."""
        if id not in self.items:
            return False
        self.items[id]["result"] = result
        self.items[id]["artifacts"] = artifacts or []
        self.save()
        return True

    def get_context_for_task(self, id: str) -> str:
        """Build context for worker from task's dependencies."""
        if id not in self.items:
            return ""

        task = self.items[id]
        context_parts = []

        # Add explicit context if set
        if task.get("context"):
            context_parts.append(task["context"])

        # Add results from dependencies
        for dep_id in task.get("dependencies", []):
            dep = self.items.get(dep_id)
            if dep and dep.get("result"):
                context_parts.append(f"## Result from '{dep['content']}':\n{dep['result']}")

        return "\n\n".join(context_parts)

    def enrich_dependent_tasks(self, completed_id: str) -> None:
        """After a task completes, enrich dependent task descriptions with discoveries."""
        completed = self.items.get(completed_id)
        if not completed or not completed.get("result"):
            return

        result = completed["result"]
        artifacts = completed.get("artifacts", [])

        # Extract file discoveries from result
        import re
        # Find .py, .md, .js, .ts files mentioned
        file_pattern = r'\b([\w./]+\.(?:py|md|js|ts|go|yaml|json))\b'
        files_found = list(set(re.findall(file_pattern, result)))

        # Also include artifacts
        files_found.extend(artifacts)
        files_found = list(set(files_found))

        if not files_found:
            return

        # Update dependent tasks that are still pending
        for task_id, task in self.items.items():
            if task["status"] != "pending":
                continue
            if completed_id not in task.get("dependencies", []):
                continue

            # Enrich the task content with discovered files
            content = task["content"]
            # Only enrich if task mentions reading/analyzing and doesn't already have specific files
            if any(word in content.lower() for word in ["read", "analyze", "examine", "find"]):
                if not any(f in content for f in files_found[:3]):
                    # Add key files to the task description
                    key_files = ", ".join(files_found[:5])
                    enriched = f"{content}\n[Key files from dependencies: {key_files}]"
                    self.items[task_id]["content"] = enriched

        self.save()

    def has_pending_tasks(self) -> bool:
        """Check if there are any pending tasks."""
        return any(item["status"] == "pending" for item in self.items.values())

    def save(self):
        """Save to JSON file."""
        self.path.write_text(json.dumps(self.items, indent=2))

    def load(self):
        """Load from JSON file."""
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text())
            except:
                self.items = {}

    def format_for_prompt(self) -> str:
        """Format todos for LLM context."""
        if not self.items:
            return "No todos."

        lines = ["Current todos:"]
        for item in self.list():
            status = item["status"]
            marker = {"pending": "[ ]", "in_progress": "[>]", "complete": "[x]", "blocked": "[!]"}.get(status, "[ ]")
            tier = item.get("tier", "medium")
            deps = item.get("dependencies", [])
            dep_str = f" [deps: {','.join(deps)}]" if deps else ""
            lines.append(f"  {marker} {item['id']} ({tier}): {item['content']}{dep_str}")

            # Show result summary for completed tasks
            if status == "complete" and item.get("result"):
                result_preview = item["result"][:100] + "..." if len(item.get("result", "")) > 100 else item.get("result", "")
                lines.append(f"      → {result_preview}")

        return "\n".join(lines)
