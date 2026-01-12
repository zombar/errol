"""Worker - executes single tasks with focused context."""
import json
import time
import sys
import select
import threading
from pathlib import Path
from typing import Optional

from llm import OllamaClient
from todos import TodoManager
from tools import execute_tool, get_tool_schemas, show_diff

# ANSI colors
DIM = "\033[2m"
RESET = "\033[0m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

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
            print(f"\r{DIM}thinking... {elapsed:.1f}s{RESET} ", end="", flush=True)
            time.sleep(0.1)

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        elapsed = time.time() - self.start_time
        # Clear the timer line
        print("\r" + " " * 30 + "\r", end="", flush=True)
        return elapsed


class SessionState:
    """Track what's been done during task execution to prevent loops."""
    def __init__(self):
        self.files_read: dict[str, str] = {}  # path -> full content (for re-serving)
        self.files_summary: dict[str, str] = {}  # path -> summary (for session state display)
        self.globs_cache: dict[str, str] = {}  # "pattern|path" -> result
        self.globs_done: list[str] = []
        self.commands_run: list[str] = []

    def record_read(self, path: str, content: str):
        # Store full content for re-serving with different offsets
        self.files_read[path] = content
        # Store summary for session state display
        lines = content.split('\n')
        self.files_summary[path] = f"{len(lines)} lines"

    def record_glob(self, pattern: str, path: str, results: str):
        cache_key = f"{pattern}|{path}"
        self.globs_cache[cache_key] = results
        self.globs_done.append(f"{pattern} -> {results[:100]}")

    def get_cached_glob(self, pattern: str, path: str) -> Optional[str]:
        cache_key = f"{pattern}|{path}"
        return self.globs_cache.get(cache_key)

    def record_bash(self, command: str):
        self.commands_run.append(command[:100])

    def invalidate(self, path: str):
        """Remove a file from cache (after edit/write)."""
        if path in self.files_read:
            del self.files_read[path]
        if path in self.files_summary:
            del self.files_summary[path]

    def get_cached_slice(self, path: str, offset: int, limit: int) -> Optional[str]:
        """Return cached content slice, or None if not cached."""
        if path not in self.files_read:
            return None
        lines = self.files_read[path].split('\n')
        selected = lines[offset:offset + limit]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)

    def get_summary(self) -> str:
        parts = []
        if self.files_summary:
            files_info = [f"{p} ({s})" for p, s in self.files_summary.items()]
            parts.append(f"Files already read: {', '.join(files_info)}")
        if self.globs_done:
            parts.append(f"Globs completed: {len(self.globs_done)}")
        if self.commands_run:
            parts.append(f"Commands run: {len(self.commands_run)}")
        return "\n".join(parts) if parts else ""


WORKER_PROMPT = """You are executing a specific task. Focus only on this task.

Task: {task_content}

{context_section}

## Available Tools (ONLY use these - no others exist):
- read_file(path, offset, limit): Read file contents
- write_file(path, content): Create or overwrite files
- edit_file(path, old_string, new_string): Replace unique string in file
- bash(command, timeout): Run shell commands
- glob(pattern, path): Find files matching pattern

IMPORTANT: These are the ONLY tools. Do not try "search", "grep", "find_file", or any other tool names.

## Guidelines
- Call ONE tool at a time and wait for the result
- Do NOT re-read files you've already read - use the content from earlier
- Never use placeholders - use actual values
- Complete the task, then provide a summary
{session_state}
When done, end with:
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

    # If stdin is not a tty (piped input), auto-confirm
    if not sys.stdin.isatty():
        return True

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        # Can't get terminal settings, auto-confirm
        return True

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


def execute_with_cache(name: str, args: dict, session: SessionState) -> str:
    """Execute tool with caching for read_file to prevent loops."""
    path = args.get("path", "")

    # Parse offset/limit for read_file
    if name == "read_file":
        offset = args.get("offset", 0) or 0
        limit = args.get("limit", 2000) or 2000
        if isinstance(offset, str):
            offset = int(offset) if offset != "null" else 0
        if isinstance(limit, str):
            limit = int(limit) if limit != "null" else 2000

        # Serve from cache if available
        if path in session.files_read:
            cached = session.get_cached_slice(path, offset, limit)
            return f"[CACHED] {cached}"

        # First read: fetch FULL file to cache, then return requested slice
        full_result = execute_tool(name, {"path": path})  # No offset/limit = full file
        if not full_result.startswith("Error"):
            session.record_read(path, full_result)
            # Return the slice that was actually requested
            return session.get_cached_slice(path, offset, limit)
        return full_result

    # Check cache for glob
    if name == "glob":
        pattern = args.get("pattern", "")
        glob_path = args.get("path", ".")
        cached_glob = session.get_cached_glob(pattern, glob_path)
        if cached_glob is not None:
            return f"[CACHED] {cached_glob}"

    result = execute_tool(name, args)

    # Invalidate cache on file modifications
    if name in ("edit_file", "write_file"):
        session.invalidate(path)

    # Record for session tracking
    if name == "glob":
        session.record_glob(args.get("pattern", ""), args.get("path", "."), result)
    elif name == "bash":
        session.record_bash(args.get("command", ""))

    return result


def execute_task(task_id: str, todos: TodoManager, client: OllamaClient,
                 models: dict, max_turns: int = 20) -> bool:
    """Execute a single task and store results."""
    task = todos.get(task_id)
    if not task:
        print(f"{DIM}Task {task_id} not found{RESET}")
        return False

    if task["status"] == "complete":
        print(f"{DIM}Task {task_id} already complete{RESET}")
        return True

    # Mark as in progress
    todos.start(task_id)

    # Get appropriate model for tier
    tier = task.get("tier", "medium")
    model = models.get(tier, models.get("medium"))

    print(f"\n{MAGENTA}▶ errol{RESET} {DIM}worker mode{RESET}")
    print(f"{DIM}Executing: {task['content']}{RESET}")
    print(f"{DIM}Using {model} ({tier}){RESET}")

    # Initialize session state for tracking
    session = SessionState()

    # Build context from dependencies
    context = todos.get_context_for_task(task_id)
    context_section = f"Context from previous tasks:\n{context}" if context else ""

    def build_system_prompt():
        session_summary = session.get_summary()
        session_state = f"\n## Session State\n{session_summary}\n" if session_summary else ""
        return WORKER_PROMPT.format(
            task_content=task["content"],
            context_section=context_section,
            session_state=session_state
        )

    system = build_system_prompt()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Execute this task: {task['content']}"}
    ]

    tools = get_tool_schemas()
    full_output = []

    timer = ThinkingTimer()

    for turn in range(max_turns):
        # Update system prompt with current session state
        messages[0]["content"] = build_system_prompt()

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
        print(f"\n{MAGENTA}errol{RESET} {DIM}({elapsed:.1f}s){RESET}")
        if content:
            print(f"{DIM}{content}{RESET}")
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
                    result = execute_with_cache(name, args, session)
                    print(f"{GREEN}✓{RESET} {DIM}{result[:500]}{'...' if len(result) > 500 else ''}{RESET}")
                else:
                    result = "Tool execution skipped by user"
                    print(f"{YELLOW}○ skipped{RESET}")

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

    # Enrich dependent tasks with discoveries from this task
    todos.enrich_dependent_tasks(task_id)

    print(f"\n{GREEN}✓{RESET} {DIM}Task complete: {task_id}{RESET}")
    if artifacts:
        print(f"{DIM}Artifacts: {', '.join(artifacts)}{RESET}")

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
