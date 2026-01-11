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

from llm import OllamaClient
from router import Router
from tools import execute_tool, get_tool_schemas, TOOLS
from todos import TodoManager

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

Guidelines:
- Read files before editing them
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
    """Ask user to select a model."""
    # Get suggested model from router
    suggested, category = router.classify(task)
    models = config["models"]

    print(f"\nSelect model (suggested: {suggested} for '{category}'):")
    print(f"  [1] {models['small']} (small - fast)")
    print(f"  [2] {models['medium']} (medium)")
    print(f"  [3] {models['large']} (large - powerful)")
    print(f"  [Enter] Use suggested")

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
    else:
        return suggested

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
        # Get response with streaming and timer
        full_content = ""
        tool_calls = []
        first_token = True

        timer.start()

        for chunk in client.chat(model, messages, tools=tools, stream=True):
            if first_token:
                timer.stop()
                print("[errol] ", end="", flush=True)
                first_token = False

            msg = chunk.get("message", {})

            # Stream text content
            if "content" in msg and msg["content"]:
                print(msg["content"], end="", flush=True)
                full_content += msg["content"]

            # Collect tool calls
            if "tool_calls" in msg:
                tool_calls.extend(msg["tool_calls"])

        if first_token:  # No response received
            timer.stop()

        print()  # Newline after response

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
    action: str = typer.Argument(..., help="add|list|done|start|rm|clear"),
    content: Optional[str] = typer.Argument(None, help="Todo content or ID")
):
    """Manage todos: add, list, done, start, rm, clear"""
    todos = TodoManager()

    if action == "add":
        if not content:
            print("Usage: errol todo add 'task description'")
            return
        id = todos.add(content)
        print(f"Added todo {id}: {content}")

    elif action == "list":
        items = todos.list(status=content)  # content can be status filter
        if not items:
            print("No todos.")
        else:
            for item in items:
                status = item["status"]
                marker = {"pending": "[ ]", "in_progress": "[>]", "complete": "[x]"}[status]
                print(f"{marker} {item['id']}: {item['content']}")

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

    else:
        print(f"Unknown action: {action}")
        print("Actions: add, list, done, start, rm, clear")


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


if __name__ == "__main__":
    app()
