#!/usr/bin/env python3
"""Errol - A self-modifying MoE local LLM agent."""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import sys
import json
import yaml
import time
import threading
import typer
from pathlib import Path
from typing import Optional

import re
import select
from llm import OllamaClient

# Read-only tools that auto-execute after countdown
READONLY_TOOLS = ("read_file", "glob")
from router import Router
from tools import execute_tool, get_tool_schemas, TOOLS
from todos import TodoManager
from orchestrator import plan_task, display_plan, confirm_plan, edit_plan
from worker import execute_task, work_all


def extract_json_objects(text: str) -> list[str]:
    """Extract JSON objects from text, handling nested braces."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Found start of object, count braces to find end
            depth = 0
            start = i
            in_string = False
            escape = False
            while i < len(text):
                ch = text[i]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            objects.append(text[start:i+1])
                            break
                i += 1
        i += 1
    return objects


def parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract first valid tool call from JSON in text (fallback for models without native tool support)."""
    # Extract all JSON objects from text
    candidates = extract_json_objects(text)

    for match in candidates:
        try:
            obj = json.loads(match)
            # Accept both "arguments" and "parameters" keys
            args = obj.get("arguments") or obj.get("parameters")
            if "name" in obj and args:
                # Skip if arguments contain placeholders
                args_str = json.dumps(args)
                if '<' in args_str and '>' in args_str:
                    continue
                # Return first valid tool call only
                return [{
                    "function": {
                        "name": obj["name"],
                        "arguments": args
                    }
                }]
        except json.JSONDecodeError:
            continue

    return []

app = typer.Typer(help="Errol - MoE local LLM coding agent")

class Timer:
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

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        # Clear the timer line
        print("\r" + " " * 30 + "\r", end="", flush=True)

# Self-awareness: know where our code lives
SELF_PATH = Path(__file__).parent.resolve()

SYSTEM_PROMPT = """You are Errol, a helpful coding and devops assistant.

You have access to these tools:
- read_file: Read file contents
- write_file: Write/create files
- edit_file: Edit files (find/replace unique strings)
- bash: Run shell commands
- glob: Find files by pattern

IMPORTANT - How to use tools:
- Call ONE tool at a time and wait for the result before calling the next
- Never use placeholders like <file-path> - use actual values from previous results
- After each tool call, you will receive the result, then decide what to do next

Guidelines:
- ALWAYS read files before editing or overwriting them
- Before using write_file, check if the file exists with glob or read_file first
- Make minimal, focused changes
- Test changes when possible (use bash to run tests/compilers)

Self-modification:
- Your source code is at: {self_path}
- You can modify your own code using edit_file
- After self-edits, validate with: python -m py_compile <file>
- Tell the user to restart after self-modifications

{todos}
"""

def load_config() -> dict:
    """Load config from yaml file."""
    config_path = SELF_PATH / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    return {
        "ollama": {"host": "http://localhost:11434", "timeout": 300},
        "models": {"small": "qwen2.5:3b", "medium": "qwen2.5:7b", "large": "qwen2.5-coder:14b"},
        "agent": {"max_turns": 20}
    }

def select_model(config: dict, router: Router, task: str, available: list) -> str:
    """Ask user to select a model with countdown auto-select."""
    import termios
    import tty

    # Get suggested model from router
    suggested, category = router.classify(task)
    models = config["models"]

    print(f"\nSelect model (suggested: {suggested} for '{category}'):")
    print(f"  [1] {models['small']} (small - fast)")
    print(f"  [2] {models['medium']} (medium)")
    print(f"  [3] {models['large']} (large - powerful)")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        for i in range(3, 0, -1):
            print(f"\rChoice [1/2/3]: auto-select in {i}s ", end="", flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                ch = sys.stdin.read(1)
                print()  # newline
                if ch == "1":
                    return models["small"]
                elif ch == "2":
                    return models["medium"]
                elif ch == "3":
                    return models["large"]
                else:
                    return suggested
        print("\r" + " " * 40 + "\r", end="", flush=True)
        return suggested
    except (termios.error, OSError):
        # Fallback for non-TTY (e.g., piped input)
        try:
            choice = input("Choice [1/2/3/Enter]: ").strip()
        except EOFError:
            choice = ""
        if choice == "1":
            return models["small"]
        elif choice == "2":
            return models["medium"]
        elif choice == "3":
            return models["large"]
        return suggested
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (termios.error, OSError):
            pass

def preview_file_change(name: str, args: dict) -> str:
    """Generate a diff preview for file operations."""
    from tools import show_diff
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

def countdown_confirm(seconds: int = 3) -> bool:
    """Auto-confirm after countdown, allow 'n' to cancel."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)  # Don't wait for Enter
        for i in range(seconds, 0, -1):
            print(f"\rAuto-execute in {i}s [n to cancel, Enter to run now] ", end="", flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                ch = sys.stdin.read(1).lower()
                print()  # newline
                return ch != 'n'
        print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear line
        return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def confirm_tool(name: str, args: dict) -> bool:
    """Ask user to confirm tool execution, showing diff for file changes."""
    print(f"\n{'='*60}")
    print(f"Tool: {name}")

    # Show diff preview for file operations
    if name in ("edit_file", "write_file"):
        diff = preview_file_change(name, args)
        if diff:
            print(f"{'='*60}")
            print(diff)
        else:
            # Show args if no diff (new file or error)
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

    # Auto-confirm read-only tools with countdown
    if name in READONLY_TOOLS:
        return countdown_confirm()

    try:
        confirm = input("Execute? [y/N]: ").strip().lower()
    except EOFError:
        confirm = "n"

    return confirm in ("y", "yes")

def agent_loop(task: str, config: dict, todos: TodoManager):
    """Main agent execution loop."""
    client = OllamaClient(
        host=config["ollama"]["host"],
        timeout=config["ollama"]["timeout"]
    )
    router = Router(client, config["models"])

    # Check Ollama is running
    if not client.is_available():
        print("Error: Ollama not available. Start it with: ollama serve")
        return

    # Check configured models are available
    available = client.list_models()
    configured = set(config["models"].values())
    missing = configured - set(available)
    if missing:
        print(f"Warning: Missing models: {', '.join(missing)}")
        print(f"Pull with: ollama pull <model>")
        print(f"Available: {', '.join(sorted(available))}")
        return

    # Let user select model
    model = select_model(config, router, task, available)
    print(f"\n[errol] Using {model}")

    # Build system prompt with self-awareness
    system = SYSTEM_PROMPT.format(
        self_path=SELF_PATH,
        todos=todos.format_for_prompt()
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task}
    ]

    tools = get_tool_schemas()
    max_turns = config["agent"]["max_turns"]
    timer = Timer()

    for turn in range(max_turns):
        # Get response (non-streaming for reliable tool calls)
        full_content = ""
        tool_calls = []

        timer.start()
        response = client.chat_sync(model, messages, tools=tools)
        timer.stop()
        elapsed = time.time() - timer.start_time

        msg = response.get("message", {})
        full_content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Fallback: parse tool calls from text if model doesn't use native format
        if not tool_calls and full_content:
            tool_calls = parse_tool_calls_from_text(full_content)

        print(f"[errol {elapsed:.1f}s] {full_content}")

        # Add assistant message to history
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Execute tool calls with confirmation
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})

                # Parse args if string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                # Ask for confirmation
                if confirm_tool(name, args):
                    result = execute_tool(name, args)
                    print(f"[result] {result[:500]}{'...' if len(result) > 500 else ''}")
                else:
                    result = "Tool execution skipped by user"
                    print(f"[skipped]")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result
                })
        else:
            # No tool calls = done
            break

    if turn >= max_turns - 1:
        print(f"\n[errol] Reached max turns ({max_turns})")


@app.command()
def chat(task: Optional[str] = typer.Argument(None, help="Task to perform")):
    """Chat with Errol. Interactive mode if no task given."""
    config = load_config()
    todos = TodoManager()

    if task:
        agent_loop(task, config, todos)
    else:
        # Interactive mode
        print("Errol - MoE coding agent. Type 'quit' to exit.\n")
        while True:
            try:
                task = input("you> ").strip()
                if not task:
                    continue
                if task.lower() in ("quit", "exit", "q"):
                    break
                agent_loop(task, config, todos)
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except EOFError:
                break


@app.command()
def todo(
    action: str = typer.Argument(..., help="add|list|show|done|start|rm|clear|clearall"),
    content: Optional[str] = typer.Argument(None, help="Todo content or ID"),
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Task tier: small|medium|large")
):
    """Manage todos: add, list, show, done, start, rm, clear, clearall"""
    todos = TodoManager()

    if action == "add":
        if not content:
            print("Usage: errol todo add 'task description' [--tier small|medium|large]")
            return
        task_tier = tier or "medium"
        id = todos.add(content, tier=task_tier)
        print(f"Added todo {id} ({task_tier}): {content}")

    elif action == "list":
        items = todos.list(status=content)  # content can be status filter
        if not items:
            print("No todos.")
        else:
            for item in items:
                status = item["status"]
                marker = {"pending": "[ ]", "in_progress": "[>]", "complete": "[x]", "blocked": "[!]"}.get(status, "[ ]")
                task_tier = item.get("tier", "medium")
                tier_marker = {"small": "S", "medium": "M", "large": "L"}.get(task_tier, "M")
                deps = item.get("dependencies", [])
                dep_str = f" [deps: {','.join(deps)}]" if deps else ""
                print(f"{marker} [{tier_marker}] {item['id']}: {item['content']}{dep_str}")

    elif action == "show":
        if not content:
            print("Usage: errol todo show <id>")
            return
        item = todos.get(content)
        if not item:
            print(f"Todo {content} not found")
            return
        print(f"\nTask: {item['id']}")
        print(f"Content: {item['content']}")
        print(f"Status: {item['status']}")
        print(f"Tier: {item.get('tier', 'medium')}")
        print(f"Dependencies: {', '.join(item.get('dependencies', [])) or 'none'}")
        print(f"Created: {item.get('created', 'unknown')}")
        if item.get('completed'):
            print(f"Completed: {item['completed']}")
        if item.get('context'):
            print(f"\nContext:\n{item['context']}")
        if item.get('result'):
            print(f"\nResult:\n{item['result']}")
        if item.get('artifacts'):
            print(f"\nArtifacts: {', '.join(item['artifacts'])}")

    elif action == "done":
        if not content:
            print("Usage: errol todo done <id>")
            return
        if todos.complete(content):
            print(f"Completed {content}")
        else:
            print(f"Todo {content} not found")

    elif action == "start":
        if not content:
            print("Usage: errol todo start <id>")
            return
        if todos.start(content):
            print(f"Started {content}")
        else:
            print(f"Todo {content} not found")

    elif action == "rm":
        if not content:
            print("Usage: errol todo rm <id>")
            return
        if todos.remove(content):
            print(f"Removed {content}")
        else:
            print(f"Todo {content} not found")

    elif action == "clear":
        count = todos.clear_completed()
        print(f"Cleared {count} completed todos")

    elif action == "clearall":
        count = todos.clear_all()
        print(f"Cleared all {count} todos")

    else:
        print(f"Unknown action: {action}")
        print("Actions: add, list, show, done, start, rm, clear, clearall")


@app.command()
def models():
    """List available Ollama models."""
    config = load_config()
    client = OllamaClient(host=config["ollama"]["host"])

    if not client.is_available():
        print("Error: Ollama not available")
        return

    print("Available models:")
    for m in client.list_models():
        print(f"  - {m}")

    print("\nConfigured models:")
    for tier, model in config["models"].items():
        print(f"  {tier}: {model}")


@app.command()
def self_check():
    """Validate Errol's own Python files."""
    import py_compile
    files = list(SELF_PATH.glob("*.py"))
    errors = []

    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
            print(f"  OK: {f.name}")
        except py_compile.PyCompileError as e:
            print(f"  ERROR: {f.name}: {e}")
            errors.append(f)

    if errors:
        print(f"\n{len(errors)} file(s) have errors")
        sys.exit(1)
    else:
        print(f"\nAll {len(files)} files OK")


@app.command()
def plan(task: str = typer.Argument(..., help="Task to break down into subtasks")):
    """Plan a complex task using orchestrator (small model), then execute."""
    config = load_config()
    todos = TodoManager()
    client = OllamaClient(
        host=config["ollama"]["host"],
        timeout=config["ollama"]["timeout"]
    )

    if not client.is_available():
        print("Error: Ollama not available. Start it with: ollama serve")
        return

    # Always clear existing tasks before new plan
    if todos.items:
        todos.clear_all()

    # Use small model to plan
    small_model = config["models"]["small"]
    print(f"\n[orchestrator] Planning with {small_model}...")

    tasks = plan_task(task, todos, client, small_model)

    if not tasks:
        print("[orchestrator] No tasks generated. Try rephrasing your request.")
        return

    display_plan(tasks, todos)

    # User review
    while True:
        try:
            choice = input("[e]dit  [a]pprove  [c]ancel: ").strip().lower()
        except EOFError:
            choice = "c"

        if choice in ("a", "approve", ""):
            break
        elif choice in ("e", "edit"):
            edit_plan(todos)
            display_plan(todos.list(status="pending"), todos)
        elif choice in ("c", "cancel"):
            todos.clear_all()
            print("[orchestrator] Cancelled.")
            return
        else:
            print("Unknown choice")

    # Auto-execute
    print("\n[orchestrator] Starting execution...\n")
    completed = work_all(todos, client, config["models"])
    print(f"\n[orchestrator] Completed {completed} task(s).")


@app.command()
def work(
    task_id: Optional[str] = typer.Argument(None, help="Specific task ID to execute"),
    all_tasks: bool = typer.Option(False, "--all", "-a", help="Execute all ready tasks")
):
    """Execute pending tasks (workers use appropriate model per task tier)."""
    config = load_config()
    todos = TodoManager()
    client = OllamaClient(
        host=config["ollama"]["host"],
        timeout=config["ollama"]["timeout"]
    )

    if not client.is_available():
        print("Error: Ollama not available. Start it with: ollama serve")
        return

    if task_id:
        # Execute specific task
        task = todos.get(task_id)
        if not task:
            print(f"Task {task_id} not found")
            return
        execute_task(task_id, todos, client, config["models"])
    elif all_tasks:
        # Execute all ready tasks
        completed = work_all(todos, client, config["models"])
        print(f"\n[worker] Completed {completed} task(s).")
    else:
        # Execute next ready task
        task = todos.get_next_task()
        if not task:
            print("No ready tasks. Use 'errol todo list' to see all tasks.")
            return
        execute_task(task["id"], todos, client, config["models"])


if __name__ == "__main__":
    app()
