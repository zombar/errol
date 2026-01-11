"""Simple todo manager with JSON persistence."""
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

class TodoManager:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path.home() / ".errol_todos.json"
        self.items: dict[str, dict] = {}
        self.load()

    def add(self, content: str) -> str:
        """Add a todo, return its ID."""
        id = uuid.uuid4().hex[:8]
        self.items[id] = {
            "id": id,
            "content": content,
            "status": "pending",
            "created": datetime.now().isoformat()
        }
        self.save()
        return id

    def list(self, status: Optional[str] = None) -> list[dict]:
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
            marker = {"pending": "[ ]", "in_progress": "[>]", "complete": "[x]"}[status]
            lines.append(f"  {marker} {item['id']}: {item['content']}")
        return "\n".join(lines)
