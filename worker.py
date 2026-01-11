"""Worker - executes single tasks with focused context."""
import json
import time
import sys
import select
import threading
from pathlib import Path

from llm import OllamaClient
from todos import TodoManager
from tools import execute_tool, get_tool_schemas, show_diff

# ANSI colors
DIM = "\033[2m"
RESET = "\033[0m"

# Read-only tools that auto-execute
READONLY_TOOLS = ("read_file", "glob")


class ThinkingTimer:
    """Background timer that shows elapsed time during inference."""
    def __init__(self):
        self.running = False
        self.thread = None
        self.start_time = 0

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            elapsed = time.time() - self.start_time
            print(f"\r[thinking... {elapsed:.1f}s] ", end="", flush=True)
            time.sleep(0.1)

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        elapsed = time.time() - self.start_time
        # Clear the timer line
        print("\r" + " " * 30 + "\r", end="", flush=True)
        return elapsed

WORKER_PROMPT = """You are executing a specific task. Focus only on this task.

Task: {task_content}

{context_section}

You have these tools:
- read_file: Read file contents
- write_file: Write/create files
- edit_file: Edit files (find/replace unique strings)
- bash: Run shell commands
- glob: Find files by pattern

Guidelines:
- Call ONE tool at a time and wait for the result
- Never use placeholders - use actual values
- Complete the task, then provide a summary of what you did

When done, end your response with:
RESULT: <brief summary of what was accomplished>
ARTIFACTS: <comma-separated list of files created/modified, or "none">
"""


def parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract first valid tool call from JSON in text."""
    import re

    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text)

    # Also try to find bare JSON objects
    if not matches:
        bare_json_pattern = r'(\{\s*"name"\s*:[\s\S]*?"arguments"\s*:[\s\S]*?\})'
        matches = re.findall(bare_json_pattern, text)

    for match in matches:
        try:
            obj = json.loads(match)
            if "name" in obj and "arguments" in obj:
                args = obj["arguments"]
                # Skip if arguments contain placeholders
                args_str = json.dumps(args)
                if '<' in args_str and '>' in args_str:
                    continue
                return [{
                    "function": {
                        "name": obj["name"],
                        "arguments": args
                    }
                }]
        except json.JSONDecodeError:
            continue

    return []


def extract_result(content: str) -> tuple[str, list[str]]:
    """Extract RESULT and ARTIFACTS from worker output."""
    result = ""
    artifacts = []

    for line in content.split("\n"):
        if line.startswith("RESULT:"):
            result = line[7:].strip()
        elif line.startswith("ARTIFACTS:"):
            artifacts_str = line[10:].strip()
            if artifacts_str.lower() != "none":
                artifacts = [a.strip() for a in artifacts_str.split(",") if a.strip()]

    # If no explicit RESULT, use last non-empty paragraph as result
    if not result:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if paragraphs:
            result = paragraphs[-1][:500]

    return result, artifacts


def countdown_confirm(seconds: int = 3) -> bool:
    """Auto-confirm after countdown, allow 'n' to cancel."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        for i in range(seconds, 0, -1):
            print(f"\rAuto-execute in {i}s [n to cancel, Enter to run now] ", end="", flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                ch = sys.stdin.read(1).lower()
                print()
                return ch != 'n'
        print("\r" + " " * 50 + "\r", end="", flush=True)
        return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def preview_file_change(name: str, args: dict) -> str:
    """Generate a diff preview for file operations."""
    path = args.get("path", "")
    p = Path(path).expanduser().resolve()

    if name == "edit_file" and p.exists():
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        content = p.read_text()
        if content.count(old_string) == 1:
            new_content = content.replace(old_string, new_string)
            return show_diff(content, new_content, p.name)
    elif name == "write_file" and p.exists():
        old_content = p.read_text()
        new_content = args.get("content", "")
        return show_diff(old_content, new_content, p.name)

    return ""


def confirm_tool(name: str, args: dict) -> bool:
    """Ask user to confirm tool execution."""
    print(f"\n{'='*60}")
    print(f"Tool: {name}")

    if name in ("edit_file", "write_file"):
        diff = preview_file_change(name, args)
        if diff:
            print(f"{'='*60}")
            print(diff)
        else:
            args_preview = json.dumps(args, indent=2)
            if len(args_preview) > 500:
                args_preview = args_preview[:500] + "..."
            print(f"Args: {args_preview}")
    else:
        args_preview = json.dumps(args, indent=2)
        if len(args_preview) > 300:
            args_preview = args_preview[:300] + "..."
        print(f"Args: {args_preview}")

    print(f"{'='*60}")

    # Auto-confirm read-only tools
    if name in READONLY_TOOLS:
        return countdown_confirm()

    try:
        confirm = input("Execute? [y/N]: ").strip().lower()
    except EOFError:
        confirm = "n"

    return confirm in ("y", "yes")


def execute_task(task_id: str, todos: TodoManager, client: OllamaClient,
                 models: dict, max_turns: int = 20) -> bool:
    """Execute a single task and store results."""
    task = todos.get(task_id)
    if not task:
        print(f"[worker] Task {task_id} not found")
        return False

    if task["status"] == "complete":
        print(f"[worker] Task {task_id} already complete")
        return True

    # Mark as in progress
    todos.start(task_id)

    # Get appropriate model for tier
    tier = task.get("tier", "medium")
    model = models.get(tier, models.get("medium"))

    print(f"\n[worker] Executing: {task['content']}")
    print(f"[worker] Using {model} ({tier})")

    # Build context from dependencies
    context = todos.get_context_for_task(task_id)
    context_section = f"Context from previous tasks:\n{context}" if context else ""

    system = WORKER_PROMPT.format(
        task_content=task["content"],
        context_section=context_section
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Execute this task: {task['content']}"}
    ]

    tools = get_tool_schemas()
    full_output = []

    timer = ThinkingTimer()

    for turn in range(max_turns):
        timer.start()
        response = client.chat_sync(model, messages, tools=tools)
        elapsed = timer.stop()

        msg = response.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Fallback: parse tool calls from text
        if not tool_calls and content:
            tool_calls = parse_tool_calls_from_text(content)

        # Dim LLM output to distinguish from tool output
        print(f"[worker {elapsed:.1f}s] {DIM}{content}{RESET}")
        full_output.append(content)

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls if tool_calls else None})

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                if confirm_tool(name, args):
                    result = execute_tool(name, args)
                    print(f"[result] {result[:500]}{'...' if len(result) > 500 else ''}")
                else:
                    result = "Tool execution skipped by user"
                    print("[skipped]")

                messages.append({"role": "tool", "content": result})
        else:
            # No tool calls = done
            break

    # Extract and store result
    combined_output = "\n".join(full_output)
    result_summary, artifacts = extract_result(combined_output)

    # If no explicit result, store full output
    if not result_summary:
        result_summary = combined_output

    todos.set_result(task_id, result_summary, artifacts)
    todos.complete(task_id)

    print(f"\n[worker] Task complete: {task_id}")
    if artifacts:
        print(f"[worker] Artifacts: {', '.join(artifacts)}")

    return True


def work_all(todos: TodoManager, client: OllamaClient, models: dict) -> int:
    """Execute all ready tasks in sequence."""
    completed = 0

    while True:
        task = todos.get_next_task()
        if not task:
            break

        if execute_task(task["id"], todos, client, models):
            completed += 1

    return completed
