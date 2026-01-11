"""Orchestrator - uses small model to plan and coordinate tasks."""
import json
import re
from llm import OllamaClient
from todos import TodoManager

PLANNING_PROMPT = """You are a task planner. Break down the user's request into subtasks.

For each subtask, output a JSON object on its own line:
{{"action": "add_task", "content": "task description", "tier": "small|medium|large", "depends_on": []}}

Tier guidelines:
- small: File listing, glob, simple reads, quick checks
- medium: Reading and analyzing files, explanations
- large: Writing code, generating content, complex edits

The depends_on field should list indices (0-based) of tasks that must complete first.

IMPORTANT: For documentation tasks, always plan to:
1. First discover ALL source files in the repo (*.py, *.js, *.ts, *.go, etc.)
2. Read the actual source code files to understand what they do
3. Then generate documentation based on the code content

After all tasks, output:
{{"action": "done"}}

Example for "document this repo":
{{"action": "add_task", "content": "Find all source files (*.py, *.js, *.ts, etc.)", "tier": "small", "depends_on": []}}
{{"action": "add_task", "content": "Read and analyze the main entry point and core modules", "tier": "medium", "depends_on": [0]}}
{{"action": "add_task", "content": "Read and analyze utility/helper modules", "tier": "medium", "depends_on": [0]}}
{{"action": "add_task", "content": "Write comprehensive documentation with actual content to DOCS.md", "tier": "large", "depends_on": [1, 2]}}
{{"action": "done"}}

Now break down this request:
{task}
"""


def plan_task(task: str, todos: TodoManager, client: OllamaClient, model: str) -> list[dict]:
    """Use small model to break task into subtasks."""
    messages = [
        {"role": "user", "content": PLANNING_PROMPT.format(task=task)}
    ]

    response = client.chat_sync(model, messages)
    content = response.get("message", {}).get("content", "")

    # Parse the JSON actions from response
    created_tasks = []
    task_ids = []  # Map index to task ID

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try to extract JSON from line
        match = re.search(r'\{[^}]+\}', line)
        if not match:
            continue

        try:
            action = json.loads(match.group())
        except json.JSONDecodeError:
            continue

        if action.get("action") == "add_task":
            # Convert depends_on indices to actual task IDs
            dep_indices = action.get("depends_on", [])
            dependencies = [task_ids[i] for i in dep_indices if i < len(task_ids)]

            task_id = todos.add(
                content=action.get("content", ""),
                tier=action.get("tier", "medium"),
                dependencies=dependencies
            )
            task_ids.append(task_id)
            created_tasks.append(todos.get(task_id))

        elif action.get("action") == "done":
            break

    return created_tasks


def display_plan(tasks: list[dict], todos: TodoManager) -> None:
    """Display the planned tasks for user review."""
    if not tasks:
        print("\n[orchestrator] No tasks created.")
        return

    print("\n[orchestrator] Proposed tasks:\n")
    for task in tasks:
        deps = task.get("dependencies", [])
        dep_str = ""
        if deps:
            dep_contents = [todos.get(d)["content"][:30] + "..." if len(todos.get(d)["content"]) > 30 else todos.get(d)["content"] for d in deps if todos.get(d)]
            dep_str = f"\n      depends on: {', '.join(dep_contents)}"

        tier_marker = {"small": "S", "medium": "M", "large": "L"}.get(task.get("tier", "medium"), "M")
        print(f"  [{tier_marker}] {task['id']}: {task['content']}{dep_str}")
    print()


def confirm_plan() -> bool:
    """Ask user to confirm the plan."""
    try:
        choice = input("[e]dit  [a]pprove  [c]ancel: ").strip().lower()
        return choice in ("a", "approve", "")
    except EOFError:
        return False


def edit_plan(todos: TodoManager) -> None:
    """Interactive plan editing."""
    while True:
        print("\nEdit commands:")
        print("  tier <id> <small|medium|large> - Change task tier")
        print("  edit <id> <new content>        - Edit task content")
        print("  rm <id>                        - Remove task")
        print("  add <content>                  - Add new task")
        print("  deps <id> <dep_id1,dep_id2>    - Set dependencies")
        print("  done                           - Finish editing")

        try:
            cmd = input("\nedit> ").strip()
        except EOFError:
            break

        if not cmd or cmd == "done":
            break

        parts = cmd.split(maxsplit=2)
        action = parts[0] if parts else ""

        if action == "tier" and len(parts) >= 3:
            todos.update(parts[1], tier=parts[2])
            print(f"Updated tier for {parts[1]}")

        elif action == "edit" and len(parts) >= 3:
            todos.update(parts[1], content=parts[2])
            print(f"Updated content for {parts[1]}")

        elif action == "rm" and len(parts) >= 2:
            todos.remove(parts[1])
            print(f"Removed {parts[1]}")

        elif action == "add" and len(parts) >= 2:
            content = " ".join(parts[1:])
            tier = input("Tier [small/medium/large]: ").strip() or "medium"
            task_id = todos.add(content, tier=tier)
            print(f"Added {task_id}")

        elif action == "deps" and len(parts) >= 3:
            dep_ids = [d.strip() for d in parts[2].split(",")]
            todos.update(parts[1], dependencies=dep_ids)
            print(f"Updated dependencies for {parts[1]}")

        else:
            print("Unknown command")

        # Show current state
        display_plan(todos.list(status="pending"), todos)
