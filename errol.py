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
from context_tracker import ContextTracker

# ANSI colors for styled output
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"

# Readline markers for non-printing characters (fixes cursor position with colors)
RL_START = "\001"  # Start of non-printing sequence
RL_END = "\002"    # End of non-printing sequence

# Read-only tools that auto-execute after countdown
READONLY_TOOLS = ("read_file", "glob", "grep")
from tools import execute_tool, get_tool_schemas, validate_tool_call, TOOLS, _fix_indentation


def looks_like_question(text: str) -> bool:
    """Detect if the model is asking for user input instead of continuing."""
    if not text:
        return False

    text_lower = text.lower()

    # Phrases that indicate task completion (not waiting for input)
    completion_phrases = [
        "here are",
        "here is",
        "i have",
        "done",
        "complete",
        "finished",
        "created",
        "written",
        "edited",
        "the files",
        "the result",
    ]
    for phrase in completion_phrases:
        if phrase in text_lower:
            return False

    # Only trigger on direct questions requiring action
    action_phrases = [
        "which file should i",
        "which one should i",
        "what would you like",
        "what you'd like",
        "what do you want",
        "what you want",
        "let me know what",
        "let me know which",
        "do you want me to",
        "should i start",
        "should i proceed",
        "please select",
        "please choose",
        "please specify",
        "could you",
        "can you clarify",
        "can you specify",
    ]
    for phrase in action_phrases:
        if phrase in text_lower:
            return True

    return False


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

app = typer.Typer(
    name="errol",
    help="Errol - MoE local LLM coding agent",
    invoke_without_command=True,
    no_args_is_help=False
)

def interactive_input(prompt: str, prefill: str = "") -> str:
    """Get input with readline support for history and multiline paste."""
    import readline
    import sys

    # Set up prefill using readline startup hook
    if prefill:
        readline.set_startup_hook(lambda: readline.insert_text(prefill))

    try:
        lines = [input(prompt)]

        # Check for pasted multiline input - read any pending lines
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not ready:
                break
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.rstrip('\n'))

        return '\n'.join(lines).strip()
    except EOFError:
        return ""
    finally:
        readline.set_startup_hook()


@app.callback()
def main(ctx: typer.Context):
    """Errol - local LLM coding agent. Run without arguments for interactive mode."""
    if ctx.invoked_subcommand is None:
        # No command given - start interactive chat
        import readline

        config = load_config()
        last_task = ""
        prefill = ""

        print(f"\n{MAGENTA}▶ errol{RESET} {DIM}interactive mode{RESET}")
        print(f"{DIM}↑/↓ history, ←/→ cursor, 'quit' to exit{RESET}\n")

        while True:
            try:
                task = interactive_input(f"{RL_START}{CYAN}{RL_END}▷{RL_START}{RESET}{RL_END} ", prefill)
                prefill = ""  # Clear prefill after input

                if not task:
                    continue
                if task.lower() in ("quit", "exit", "q"):
                    print(f"{DIM}Bye!{RESET}")
                    break

                # Add to readline history
                readline.add_history(task)
                last_task = task

                cancelled = agent_loop(task, config)
                if cancelled:
                    prefill = last_task  # Prefill for easy retry
                print()  # Blank line between interactions

            except KeyboardInterrupt:
                # Ctrl+C at prompt - just show new prompt
                print()
                continue
            except EOFError:
                print(f"{DIM}Bye!{RESET}")
                break


class Timer:
    """Background timer that shows elapsed time during inference with spinner."""
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self):
        self.running = False
        self.thread = None
        self.start_time = 0
        self.frame = 0

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.frame = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            elapsed = time.time() - self.start_time
            spin = self.SPINNER[self.frame % len(self.SPINNER)]
            print(f"\r{DIM}{spin} thinking...{RESET} {DIM}({elapsed:.1f}s){RESET}  ", end="", flush=True)
            self.frame += 1
            time.sleep(0.08)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        # Clear the timer line
        print("\r" + " " * 40 + "\r", end="", flush=True)

# Self-awareness: know where our code lives
SELF_PATH = Path(__file__).parent.resolve()

SYSTEM_PROMPT = """You are Errol, a helpful coding and devops assistant.

Current working directory: {cwd}

## Available Tools (ONLY these exist - no others):
- read_file(path, offset, limit): Read file contents
- write_file(path, content): Create or overwrite files
- edit_file(path, old_string, new_string, replace_all=false): Find/replace string in file
- bash(command, timeout): Run shell commands
- glob(pattern, path): Find files by filename pattern (e.g. **/*.py)
- grep(pattern, path, include): Search for text inside files

IMPORTANT: These are the ONLY tools. Do not try "search", "container.exec", "repo_browser", or any other tool names.

## CRITICAL: Stay On Task
- Your ONLY job is to answer/complete what the user asked
- Do NOT provide unsolicited code reviews, analysis, or suggestions
- Do NOT explain what the code does unless asked
- After reading files, IMMEDIATELY address the user's original request
- If the user asks to add/modify something, DO that - don't analyze instead
- When the task is clear, proceed without asking "what do you want me to do"

## CRITICAL: Actually Make Changes
- To modify files, you MUST call edit_file or write_file - describing changes is NOT enough
- NEVER show a diff or code block and claim you made changes without calling a tool
- NEVER say "File updated" or "I added" unless you actually called edit_file/write_file
- If you want to change code, call the tool. Do not just show what the change would look like.

## CRITICAL: Do NOT Repeat Yourself
- NEVER re-read a file you have already read - the system will warn you
- NEVER repeat the same grep/glob search twice
- If you need to recall earlier information, check the CONTEXT REMINDER when provided
- Each tool call must make NEW progress toward the goal

## Rules:
- Call ONE tool at a time, wait for result

## Completion Requirement:
After gathering information, you MUST directly address the user's original request.
Do not summarize what you read - do what the user asked.
"""

# How often to inject context reminders (every N tool calls)
REMINDER_INTERVAL = 5

def load_config() -> dict:
    """Load config from yaml file."""
    config_path = SELF_PATH / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    return {
        "ollama": {"host": "http://localhost:11434", "timeout": 300},
        "model": "qwen2.5-coder:14b",
        "agent": {"max_turns": 20}
    }

def preview_file_change(name: str, args: dict) -> str:
    """Generate a diff preview for file operations."""
    from tools import show_diff
    import difflib
    path = args.get("path", "")
    if not path:
        return ""
    p = Path(path).expanduser().resolve()

    if name == "edit_file" and p.exists() and p.is_file():
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        if not old_string:
            return ""
        content = p.read_text()
        count = content.count(old_string)
        replace_all = args.get("replace_all", False)
        if count >= 1:
            # Fix indentation in preview to show what will actually be applied
            fixed_new = _fix_indentation(old_string, new_string)
            if replace_all:
                new_content = content.replace(old_string, fixed_new)
            else:
                # Replace only first occurrence
                new_content = content.replace(old_string, fixed_new, 1)
            diff = show_diff(content, new_content, p.name)
            if count > 1 and not replace_all:
                diff += f"\n{YELLOW}(replacing first of {count} occurrences){RESET}"
            return diff
        elif count == 0:
            # String not found - try to find close matches
            lines = content.splitlines()
            old_lines = old_string.splitlines()
            if old_lines:
                # Find best matching line
                matches = difflib.get_close_matches(old_lines[0].strip(), [l.strip() for l in lines], n=1, cutoff=0.6)
                if matches:
                    return f"{RED}String not found.{RESET} Similar line exists:\n{DIM}{matches[0]}{RESET}\n{YELLOW}Check indentation/whitespace.{RESET}"
            return f"{RED}String not found in file.{RESET} {YELLOW}Check indentation/whitespace.{RESET}"
    elif name == "write_file":
        new_content = args.get("content", "")
        if p.exists() and p.is_file():
            # Overwriting existing file
            old_content = p.read_text()
            return show_diff(old_content, new_content, p.name)
        else:
            # New file - show as all additions
            return show_diff("", new_content, p.name)

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
            print(f"\r{DIM}auto-run in {i}s{RESET} {DIM}[n=cancel, enter=run]{RESET} ", end="", flush=True)
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
    print(f"\n{DIM}{'─'*60}{RESET}")
    print(f"{CYAN}◆{RESET} {BOLD}{name}{RESET}")

    # Show diff preview for file operations
    if name in ("edit_file", "write_file"):
        diff = preview_file_change(name, args)
        if diff:
            print(f"{DIM}{'─'*60}{RESET}")
            print(diff)
        else:
            # Show args if no diff (new file or error)
            args_preview = json.dumps(args, indent=2)
            if len(args_preview) > 500:
                args_preview = args_preview[:500] + "..."
            print(f"{DIM}{args_preview}{RESET}")
    else:
        args_preview = json.dumps(args, indent=2)
        if len(args_preview) > 300:
            args_preview = args_preview[:300] + "..."
        print(f"{DIM}{args_preview}{RESET}")

    print(f"{DIM}{'─'*60}{RESET}")

    # Auto-confirm read-only tools with countdown
    if name in READONLY_TOOLS:
        return countdown_confirm()

    try:
        confirm = input(f"{YELLOW}Execute?{RESET} [y/N]: ").strip().lower()
    except EOFError:
        confirm = "n"

    return confirm in ("y", "yes")

def agent_loop(task: str, config: dict) -> bool:
    """Main agent execution loop. Returns True if cancelled by ESC."""
    client = OllamaClient(
        host=config["ollama"]["host"],
        timeout=config["ollama"]["timeout"]
    )

    # Check Ollama is running
    if not client.is_available():
        print("Error: Ollama not available. Start it with: ollama serve")
        return False

    # Check configured model is available
    model = config["model"]
    available = client.list_models()
    if model not in available:
        print(f"Warning: Model '{model}' not found")
        print(f"Pull with: ollama pull {model}")
        print(f"Available: {', '.join(sorted(available))}")
        return False

    print(f"\n{MAGENTA}▶ errol{RESET} {DIM}using {model}{RESET}")

    # Build system prompt
    system = SYSTEM_PROMPT.format(cwd=os.getcwd())

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task}
    ]

    # Initialize context tracker
    tracker = ContextTracker(task)

    tools = get_tool_schemas()
    max_turns = config["agent"]["max_turns"]
    timer = Timer()
    reprompt_count = 0
    max_reprompts = 5
    tool_call_count = 0

    for turn in range(max_turns):
        # Get response (non-streaming for reliable tool calls)
        full_content = ""
        tool_calls = []

        timer.start()
        try:
            response = client.chat_sync(model, messages, tools=tools)
        except KeyboardInterrupt:
            timer.stop()
            print(f"\n{YELLOW}⚠ Interrupted{RESET}")
            return True  # Signal cancellation
        except Exception as e:
            # If Ollama fails to parse tool calls, retry without tools
            error_str = str(e)
            if "error parsing tool call" in error_str or "500" in error_str:
                print(f"{YELLOW}⚠ Ollama tool parsing failed, retrying without tools...{RESET}")
                try:
                    response = client.chat_sync(model, messages, tools=None)
                except KeyboardInterrupt:
                    timer.stop()
                    print(f"\n{YELLOW}⚠ Interrupted{RESET}")
                    return True
                except Exception as e2:
                    timer.stop()
                    print(f"{RED}✗ API error: {e2}{RESET}")
                    break
            else:
                timer.stop()
                print(f"{RED}✗ API error: {e}{RESET}")
                break
        timer.stop()
        elapsed = time.time() - timer.start_time

        msg = response.get("message", {})
        full_content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Fallback: parse tool calls from text if model doesn't use native format
        if not tool_calls and full_content:
            tool_calls = parse_tool_calls_from_text(full_content)

        if full_content:
            # Clean up escaped newlines and format output
            cleaned = full_content.replace("\\n", "\n").replace("\\t", "\t")
            print(f"\n{MAGENTA}errol{RESET} {DIM}({elapsed:.1f}s){RESET}")
            # Render markdown with rich
            try:
                from rich.console import Console
                from rich.markdown import Markdown
                console = Console()
                md = Markdown(cleaned)
                console.print(md)
            except ImportError:
                # Fallback if rich not installed
                for line in cleaned.split("\n"):
                    print(f"  {line}")

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

                # Check for duplicate operations before validation
                if name == "read_file":
                    path = args.get("path", "")
                    offset = args.get("offset", 0) or 0
                    if tracker.is_duplicate_read(path, offset):
                        print(f"\n{DIM}{'─'*60}{RESET}")
                        print(f"{CYAN}◆{RESET} {name}")
                        print(f"{DIM}{'─'*60}{RESET}")
                        print(f"{YELLOW}⚠ Already read this file{RESET}")
                        result = f"WARNING: You already read {path}. Use the information from before. Do NOT re-read files."
                        messages.append({"role": "tool", "content": result})
                        continue

                if name in ("grep", "glob"):
                    pattern = args.get("pattern", "")
                    if tracker.is_duplicate_search(name, pattern):
                        print(f"\n{DIM}{'─'*60}{RESET}")
                        print(f"{CYAN}◆{RESET} {name}")
                        print(f"{DIM}{'─'*60}{RESET}")
                        print(f"{YELLOW}⚠ Already performed this search{RESET}")
                        result = f"WARNING: You already searched for '{pattern}'. Use the results from before. Do NOT repeat searches."
                        messages.append({"role": "tool", "content": result})
                        continue

                # Validate before prompting
                validation_error = validate_tool_call(name, args)
                if validation_error:
                    print(f"\n{DIM}{'─'*60}{RESET}")
                    print(f"{CYAN}◆{RESET} {name}")
                    print(f"{DIM}{'─'*60}{RESET}")
                    print(f"{RED}✗ Error: {validation_error}{RESET}")
                    result = f"Error: {validation_error}"
                    # For unknown tools, add forceful user message to make LLM retry correctly
                    if "Unknown tool" in validation_error:
                        messages.append({"role": "tool", "content": result})
                        messages.append({
                            "role": "user",
                            "content": f"STOP. The tool '{name}' does not exist. You MUST use one of: read_file, write_file, edit_file, bash, glob, grep. Try again with a valid tool."
                        })
                        continue  # Skip adding result again below
                elif confirm_tool(name, args):
                    result = execute_tool(name, args)
                    # Only treat as error if starts with "Error" (not if content contains it)
                    if result.startswith("Error"):
                        print(f"{RED}✗{RESET} {result[:500]}{'...' if len(result) > 500 else ''}")
                    else:
                        print(f"{GREEN}✓{RESET} {DIM}{result[:500]}{'...' if len(result) > 500 else ''}{RESET}")

                    # Track the tool result
                    tracker.record_tool_result(name, args, result)
                    tool_call_count += 1
                else:
                    result = "Tool execution skipped by user"
                    print(f"{YELLOW}○ skipped{RESET}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result
                })

                # Inject context reminder periodically
                if tool_call_count > 0 and tool_call_count % REMINDER_INTERVAL == 0:
                    reminder = tracker.get_simple_reminder()
                    print(f"{DIM}📋 Injecting context reminder (turn {tool_call_count}){RESET}")
                    messages.append({
                        "role": "user",
                        "content": reminder
                    })
        else:
            # No tool calls - check if model is asking a question
            if looks_like_question(full_content) and reprompt_count < max_reprompts:
                reprompt_count += 1
                print(f"{DIM}↻ continuing autonomously ({reprompt_count}/{max_reprompts})...{RESET}")
                messages.append({
                    "role": "user",
                    "content": "Continue working on the task autonomously. Don't ask for input - make reasonable choices and proceed with the next step."
                })
                # Continue the loop instead of breaking
            else:
                # Truly done or max reprompts reached
                break

    if turn >= max_turns - 1:
        print(f"\n{YELLOW}⚠ Reached max turns ({max_turns}){RESET}")

    return False


@app.command()
def chat(task: Optional[str] = typer.Argument(None, help="Task to perform")):
    """Chat with Errol. Interactive mode if no task given."""
    config = load_config()

    if task:
        agent_loop(task, config)
    else:
        # Interactive mode
        print("Errol - coding agent. Type 'quit' to exit.\n")
        while True:
            try:
                task = input("you> ").strip()
                if not task:
                    continue
                if task.lower() in ("quit", "exit", "q"):
                    break
                agent_loop(task, config)
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except EOFError:
                break


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

    print(f"\nConfigured model: {config['model']}")


@app.command()
def self_check():
    """Validate Errol's own Python files and run unit tests."""
    import py_compile

    print(f"\n{MAGENTA}▶ errol{RESET} {DIM}self-check{RESET}")

    # Step 1: Syntax check
    print(f"\n{CYAN}◆{RESET} {BOLD}Syntax Check{RESET}")
    files = list(SELF_PATH.glob("*.py"))
    syntax_errors = []

    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
            print(f"  {GREEN}✓{RESET} {DIM}{f.name}{RESET}")
        except py_compile.PyCompileError as e:
            print(f"  {RED}✗{RESET} {f.name}: {e}")
            syntax_errors.append(f)

    if syntax_errors:
        print(f"\n{RED}✗{RESET} {len(syntax_errors)} file(s) have syntax errors")
        sys.exit(1)

    # Step 2: Run unit tests
    print(f"\n{CYAN}◆{RESET} {BOLD}Unit Tests{RESET}")
    from test_tools import run_all_tests
    results = run_all_tests()

    if results.failed > 0:
        print(f"\n{RED}✗{RESET} {results.passed} passed, {results.failed} failed")
        print(f"\n{DIM}Failures:{RESET}")
        for name, msg in results.errors:
            print(f"  {RED}✗{RESET} {name}: {DIM}{msg}{RESET}")
        sys.exit(1)

    print(f"\n{GREEN}✓{RESET} All {results.passed} tests passed")


if __name__ == "__main__":
    # Set program name for help output
    sys.argv[0] = "errol"
    app()
