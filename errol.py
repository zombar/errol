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
READONLY_TOOLS = ("read_file", "glob", "grep", "todo")
from tools import execute_tool, get_tool_schemas, validate_tool_call, resolve_tool_name, TOOLS
from task_tracker import get_tracker, reset_tracker

# Store last raw LLM response for 'raw' command
_last_raw_response = {"content": "", "reasoning": ""}


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


def looks_like_code_suggestion(text: str) -> bool:
    """Detect if the model described code changes without calling a tool."""
    if not text:
        return False

    # Indicators that model is showing/describing code changes
    code_indicators = [
        '```',           # Code block
        '@@',            # Diff marker
        '+ ',            # Diff addition (with space)
        '- ',            # Diff removal (with space)
        'change this',
        'replace this',
        'modify this',
        'update this',
        'should be changed to',
        'could be changed to',
        'would look like',
        'here is the',
        'here\'s the',
        'the updated',
        'the modified',
        'the new version',
    ]

    # Action descriptions that imply edits should be made (gpt-oss pattern)
    action_indicators = [
        'delete the',
        'remove the',
        'exact change required',
        'change required',
        'after the edit',
        'should only contain',
        'this will restore',
        'this will fix',
        'need to delete',
        'need to remove',
        'need to change',
        'need to edit',
        'need to update',
    ]

    text_lower = text.lower()
    for indicator in code_indicators + action_indicators:
        if indicator in text or indicator in text_lower:
            return True

    return False


def looks_like_multi_step_plan(text: str) -> bool:
    """Detect if the model outlined a multi-step plan (not just a single change)."""
    if not text:
        return False

    import re

    # Count numbered steps (1. 2. 3. or 1) 2) 3))
    numbered_pattern = r'^\s*\d+[\.\)]\s+\S'
    numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE)
    if len(numbered_matches) >= 3:
        return True

    # Count bullet points
    bullet_pattern = r'^\s*[-•*]\s+\S'
    bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE)
    if len(bullet_matches) >= 4:
        return True

    # Multiple "step" or "change" mentions
    text_lower = text.lower()
    step_words = ['step 1', 'step 2', 'change 1', 'change 2', 'first,', 'second,', 'third,']
    step_count = sum(1 for word in step_words if word in text_lower)
    if step_count >= 2:
        return True

    # Multiple code blocks (suggests multiple changes)
    code_blocks = text.count('```')
    if code_blocks >= 4:  # 2+ complete blocks (open + close each)
        return True

    return False


def extract_reasoning_level(task: str) -> tuple[str, str]:
    """Extract reasoning level prefix from task. Returns (level, cleaned_task).

    Supports prefixes like Claude's 'ultrathink':
    - 'ultrathink:' or 'think hard:' -> high reasoning
    - 'quick:' -> low reasoning
    - default -> medium reasoning
    """
    task_lower = task.lower()
    if task_lower.startswith("ultrathink:") or task_lower.startswith("think hard:"):
        return "high", task.split(":", 1)[1].strip()
    elif task_lower.startswith("quick:"):
        return "low", task.split(":", 1)[1].strip()
    return "medium", task


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


def parse_python_tool_call(text: str) -> list[dict]:
    """Parse Python-style tool calls like todo(action='add', content='...')."""
    import ast

    results = []
    tool_names = ['todo', 'glob', 'grep', 'read_file', 'edit_file', 'write_file', 'bash']

    for tool_name in tool_names:
        # Find all occurrences of tool_name( and extract balanced parentheses
        start = 0
        while True:
            # Find tool call start
            pattern = rf'\b{tool_name}\s*\('
            match = re.search(pattern, text[start:])
            if not match:
                break

            call_start = start + match.end()  # Position after opening (
            # Find matching closing )
            depth = 1
            i = call_start
            while i < len(text) and depth > 0:
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                i += 1

            if depth == 0:
                args_str = text[call_start:i-1]

                # Parse the arguments
                args = {}
                try:
                    # Try to parse as Python dict literal
                    dict_str = '{' + args_str + '}'
                    parsed = ast.literal_eval(dict_str)
                    if isinstance(parsed, dict):
                        args = parsed
                except:
                    # Fallback: manual parsing for key='value' patterns
                    # Use [^'"]* to allow empty strings and avoid matching across quotes
                    for part in re.finditer(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str):
                        key, value = part.groups()
                        # Clean up the value
                        value = value.strip().replace('\n', ' ').replace('  ', ' ')
                        args[key] = value

                # Map common aliases
                if 'description' in args and 'content' not in args:
                    args['content'] = args.pop('description')

                if args:
                    results.append({
                        "function": {
                            "name": tool_name,
                            "arguments": args
                        }
                    })

            start = call_start

    return results

def infer_tool_from_args(obj: dict):
    """Infer tool name from argument structure when name is missing."""
    # edit_file: has path, old_string, new_string
    if "path" in obj and "old_string" in obj and "new_string" in obj:
        return "edit_file", obj
    # write_file: has path and content
    if "path" in obj and "content" in obj and "old_string" not in obj:
        return "write_file", obj
    # read_file: has path only (or with offset/limit)
    if "path" in obj and len(obj) <= 3 and all(k in ["path", "offset", "limit"] for k in obj):
        return "read_file", obj
    # grep: has pattern
    if "pattern" in obj and "old_string" not in obj:
        return "grep", obj
    # glob: has pattern but different context
    if "pattern" in obj and "path" in obj and len(obj) == 2:
        return "glob", obj
    # todo: has action
    if "action" in obj:
        return "todo", obj
    # bash: has command
    if "command" in obj:
        return "bash", obj
    return None

def parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract tool calls from text (fallback for models without native tool support)."""
    # First try Python-style calls: todo(action='add', ...)
    python_calls = parse_python_tool_call(text)
    if python_calls:
        return python_calls

    # Then try JSON format
    candidates = extract_json_objects(text)
    results = []

    # Valid tool names are alphanumeric with underscores only
    def is_valid_tool_name(name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        # Reject names with special characters (channel markers, placeholders, etc.)
        if any(c in name for c in '<>|'):
            return False
        # Must be a reasonable identifier
        return all(c.isalnum() or c == '_' for c in name)

    for match in candidates:
        try:
            obj = json.loads(match)

            # Check for explicit name field
            if "name" in obj:
                name = obj["name"]
                if not is_valid_tool_name(name):
                    continue  # Skip invalid tool names
                args = obj.get("arguments") or obj.get("parameters")
                if args:
                    args_str = json.dumps(args)
                    if '<' in args_str and '>' in args_str:
                        continue
                    results.append({
                        "function": {
                            "name": name,
                            "arguments": args
                        }
                    })
                    continue

            # Try to infer tool from argument structure
            inferred = infer_tool_from_args(obj)
            if inferred:
                tool_name, args = inferred
                args_str = json.dumps(args)
                if '<' in args_str and '>' in args_str:
                    continue
                results.append({
                    "function": {
                        "name": tool_name,
                        "arguments": args
                    }
                })

        except json.JSONDecodeError:
            continue

    return results

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


def check_existing_session() -> bool:
    """Check for existing session and prompt user."""
    session_path = Path(".errol") / "last_session.json"
    plan_path = Path(".errol/PLAN.md")

    if not session_path.exists() and not plan_path.exists():
        return False  # No existing session

    # Show current plan if exists
    if plan_path.exists():
        content = plan_path.read_text()
        print(f"\n{CYAN}◆ Found existing plan:{RESET}")
        preview = content[:500] + "..." if len(content) > 500 else content
        print(f"{DIM}{preview}{RESET}")

    try:
        response = input(f"\n{YELLOW}Continue previous session?{RESET} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = "n"

    if response in ("y", "yes"):
        return True  # Will load session
    else:
        # Clear both session and plan
        if session_path.exists():
            session_path.unlink()
        if plan_path.exists():
            plan_path.unlink()
        print(f"{DIM}Session cleared.{RESET}")
        return False


def clear_saved_session():
    """Delete any saved session state."""
    session_path = Path(".errol") / "last_session.json"
    plan_path = Path(".errol/PLAN.md")
    if session_path.exists():
        session_path.unlink()
    if plan_path.exists():
        plan_path.unlink()


def show_tasks():
    """Display current tasks in TODO format with subtask indentation."""
    tracker = get_tracker()

    # Try loading saved session if no current tasks
    if not tracker.has_tasks():
        tracker.load()

    if not tracker.has_tasks():
        print(f"{DIM}No tasks.{RESET}")
        return

    print(f"\n{CYAN}Tasks:{RESET}")
    for task in tracker.list():
        if task["status"] == "completed":
            marker = f"{GREEN}✓{RESET}"
        elif task["status"] == "in_progress":
            marker = f"{YELLOW}→{RESET}"
        else:
            marker = f"{DIM}○{RESET}"

        # Indent subtasks
        indent = "    " if task.get("parent_id") else "  "
        print(f"{indent}{marker} [{task['id']}] {task['content']}")


def delete_task(task_id: str):
    """Delete a task by ID."""
    tracker = get_tracker()

    # Load saved session if needed
    if not tracker.has_tasks():
        tracker.load()

    task = tracker.get(task_id)
    if not task:
        print(f"{RED}✗ Task '{task_id}' not found.{RESET}")
        return

    tracker.remove(task_id)
    tracker.save()  # Persist the change
    print(f"{GREEN}✓ Deleted task: {task['content'][:50]}{RESET}")


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
        print(f"{DIM}↑/↓ history, 'raw' for LLM output, 'quit' to exit{RESET}\n")

        # Check for existing session on startup
        if check_existing_session():
            tracker = get_tracker()
            tracker.load()
            print(f"{GREEN}✓ Session restored. Type 'tasks' to see progress.{RESET}\n")

        while True:
            try:
                task = interactive_input(f"{RL_START}{CYAN}{RL_END}▷{RL_START}{RESET}{RL_END} ", prefill)
                prefill = ""  # Clear prefill after input

                if not task:
                    continue

                task_lower = task.lower()

                if task_lower in ("quit", "exit", "q"):
                    print(f"{DIM}Bye!{RESET}")
                    break

                # Handle special commands
                if task_lower == "continue":
                    tracker = get_tracker()
                    if not tracker.load() or not tracker.interrupted:
                        print(f"{YELLOW}No interrupted session to continue.{RESET}")
                        continue
                    # Resume with the original task
                    original_task = tracker.original_task
                    print(f"{GREEN}✓ Resuming: {original_task[:50]}...{RESET}")
                    readline.add_history("continue")
                    cancelled = agent_loop(original_task, config, resume=True)
                    if cancelled:
                        prefill = "continue"
                    print()
                    continue

                elif task_lower == "clear":
                    clear_saved_session()
                    reset_tracker()
                    print(f"{GREEN}✓ Session cleared.{RESET}")
                    continue

                elif task_lower in ("tasks", "todo"):
                    show_tasks()
                    continue

                elif task_lower == "raw":
                    show_raw_feedback()
                    continue

                elif task_lower.startswith("delete "):
                    task_id = task.split(" ", 1)[1].strip()
                    delete_task(task_id)
                    continue

                elif task_lower == "approve":
                    tracker = get_tracker()
                    if tracker.mode == "todo":
                        subtasks = [t for t in tracker.tasks if t.get("parent_id")]

                        if not subtasks:
                            # No todos yet - extract from enriched plan
                            print(f"{CYAN}◆ Extracting tasks from plan...{RESET}")
                            cancelled = agent_loop(tracker.original_task, config, resume=True)
                            if cancelled:
                                continue
                        else:
                            # Have todos - approve them and switch to write mode
                            tracker.mode = "write"
                            tracker.save()
                            print(f"{GREEN}✓ Tasks approved. Executing...{RESET}")
                            cancelled = agent_loop(tracker.original_task, config, resume=True)
                            if cancelled:
                                continue
                    elif tracker.mode == "planning":
                        print(f"{YELLOW}Still generating plan. Please wait for completion.{RESET}")
                    elif tracker.mode == "write":
                        print(f"{YELLOW}Already in write mode.{RESET}")
                    else:
                        print(f"{YELLOW}Nothing to approve.{RESET}")
                    continue

                elif task_lower == "plan":
                    tracker = get_tracker()
                    tracker.mode = "planning"
                    tracker.save()
                    print(f"{CYAN}◆ Entering planning mode. Read-only operations only.{RESET}")
                    continue

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
Reasoning: {reasoning_level}

Current working directory: {cwd}

## Tool Definitions (use ONLY these - no others exist):

{{"name": "read_file", "parameters": {{"path": "string", "offset": "int (optional)", "limit": "int (optional)"}}}}
{{"name": "write_file", "parameters": {{"path": "string", "content": "string"}}}}
{{"name": "edit_file", "parameters": {{"path": "string", "old_string": "string", "new_string": "string"}}}}
{{"name": "bash", "parameters": {{"command": "string", "timeout": "int (optional)"}}}}
{{"name": "glob", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "grep", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "todo", "parameters": {{"action": "list|add|start|complete", "content": "string (for add)", "task_id": "string (for start/complete)", "parent_id": "string (for subtasks)", "file_path": "string (optional)", "anchor": "string (optional, function/class name)"}}}}

IMPORTANT: These are the ONLY 7 tools. Do not try "search", "container.exec", "repo_browser", or any other names.

## Multi-Step Tasks
When a task requires multiple changes (e.g. "add feature X" requiring class + usage + tests):
1. FIRST call todo with action="add" for each step you plan to take
2. Before starting each step, call todo with action="start" and the task_id
3. After completing each step, call todo with action="complete" and the task_id
This ensures you complete ALL planned changes, not just the first one.

## CRITICAL: Stay On Task
- Your ONLY job is to answer/complete what the user asked
- Do NOT provide unsolicited code reviews, analysis, or suggestions
- After reading files, IMMEDIATELY address the user's original request
- If asked to modify something, call edit_file - do not just show the change

## Rules:
- Call ONE tool at a time, wait for result
- To modify files you MUST call edit_file - describing changes is NOT enough

## Completion Requirement:
After gathering information, you MUST directly address the user's original request.
"""

# Separate, focused prompt for planning mode - only includes read-only tools
PLANNING_PROMPT = """You are Errol in PLANNING MODE. Your ONLY job is to create a structured plan.

Current working directory: {cwd}

## Available Tools (ONLY these 4):

{{"name": "glob", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "grep", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "read_file", "parameters": {{"path": "string", "offset": "int (optional)", "limit": "int (optional)"}}}}
{{"name": "todo", "parameters": {{"action": "add", "content": "string", "parent_id": "{root_id}", "file_path": "string", "anchor": "string"}}}}

## Your Task:
1. First, use glob to find existing files in the project
2. IMPORTANT: If files exist, you MUST read them with read_file BEFORE creating todos - understand existing implementation first!
3. Then, create ALL todos in a SINGLE response - call todo() multiple times, once for each step
4. Stop after creating todos - do NOT write code or suggest implementations

If the project is empty (no files found), skip exploration and immediately create todos.

## CRITICAL RULES:
- ALWAYS read existing relevant files before planning - don't ignore partial implementations!
- Keep plans CONCISE: aim for 3-5 high-level tasks, not 10+ granular steps
- Group related work into single tasks (e.g., "Implement Snake class with movement" not 4 separate tasks)
- Create ALL todos in ONE response - do not wait for feedback between todos
- Do NOT output code blocks or patches
- Each todo must have file_path (the file to modify) and anchor (function/class name)

## Example (multi-step task):
User: "Add save and load features"
You call BOTH todos in one response:
- todo(action="add", content="Add save() method", file_path="game.py", anchor="class Game", parent_id="{root_id}")
- todo(action="add", content="Add load() method", file_path="game.py", anchor="class Game", parent_id="{root_id}")

After creating ALL todos, say "Plan complete. Type 'approve' to execute."
"""

# Enriched planning prompt - generates detailed implementation plan (shot 1)
ENRICHED_PLAN_PROMPT = """You are Errol in PLANNING MODE. Your job is to create a DETAILED implementation plan.

Current working directory: {cwd}

## Available Tools (read-only):
{{"name": "glob", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "grep", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "read_file", "parameters": {{"path": "string", "offset": "int (optional)", "limit": "int (optional)"}}}}

## Your Process:
1. FIRST: Run glob(pattern="**/*.py") ONCE to check if this is an existing project
2. IF files exist: read relevant ones to understand patterns and conventions
3. IF NO files exist (greenfield project): SKIP exploration entirely and proceed to step 4
4. Output a DETAILED implementation plan in the format below

IMPORTANT: For greenfield/empty projects, do NOT keep searching for files. After ONE glob returns "No files matched", immediately output the implementation plan.

## Output Format:

# Implementation Plan: [task summary]

## Overview
Brief description of what needs to be done.

## Files to Create/Modify

### [file_path]
**Changes:**
1. [change description] - `function_name(params) -> ReturnType`
2. [change description] - `function_name(params) -> ReturnType`

(Repeat for each file)

## RULES:
- Do NOT call the todo tool - output markdown only
- Do NOT write implementation code
- Include file paths and function signatures
- Keep it concise
"""

# Todo extraction prompt - extracts todos from enriched plan (shot 2)
TODO_EXTRACTION_PROMPT = """You are extracting structured todos from an implementation plan.

## The Plan:
{enriched_plan}

## Your Task:
Read the plan and create ONE todo for each discrete change needed. For each change:

Call todo(action="add", content="...", file_path="...", anchor="...", specification="...", parent_id="{root_id}")

## Rules for good todos:
- content: Brief imperative description (e.g., "Add validate_input() method")
- file_path: The exact file path from the plan
- anchor: The function/class name where the change goes (e.g., "class UserModel" or "def process_request")
- specification: Copy the relevant details from the plan - signature, integration points, patterns to follow
- Keep todos atomic - one logical change per todo
- Order todos by dependency (earlier todos should be done first)

## Example:
For a plan section like:
  ### src/auth.py
  **Changes Required:**
  1. Add login() function
     - Signature: `def login(email: str, password: str) -> Token`
     - Integration: Import jwt from jose, call validate_password()

You would call:
todo(action="add", content="Add login function", file_path="src/auth.py", anchor="module level", specification="Signature: def login(email: str, password: str) -> Token. Import jwt from jose. Call validate_password() for password check.", parent_id="{root_id}")

Now extract ALL todos from the plan. Call todo() for each one.
After creating all todos, say "Todos extracted. Type 'approve' to execute."
"""

VALIDATION_PROMPT = """You are Errol in VALIDATION MODE. Check code for correctness.

Current working directory: {cwd}

## Task Completed
{completed_task}

## Files Modified
{modified_files}

## Your Task:
1. Read the modified files using read_file
2. Check for:
   - Syntax errors (missing brackets, invalid syntax)
   - Type issues (wrong argument types, undefined variables)
   - Logical errors (off-by-one, null checks, infinite loops)
   - Import/dependency issues
3. Run tests or linters with bash if helpful (e.g., python -m py_compile file.py)

## Available Tools (ONLY these 4):
{{"name": "read_file", "parameters": {{"path": "string", "offset": "int (optional)", "limit": "int (optional)"}}}}
{{"name": "bash", "parameters": {{"command": "string", "timeout": "int (optional)"}}}}
{{"name": "grep", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}
{{"name": "glob", "parameters": {{"pattern": "string", "path": "string (optional)"}}}}

## Output Format:
After analysis, clearly state ONE of:
- ISSUES FOUND: [list issues with file:line references]
- NO ISSUES FOUND

Do NOT suggest fixes yet - just identify issues. The user will decide whether to fix them.
"""

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
        from tools import _find_match, _adjust_replacement_indent
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "") or ""
        replace_all = args.get("replace_all", False)
        line_start = args.get("line_start")
        line_end = args.get("line_end")

        content = p.read_text()

        # Handle line-based selection - direct replacement without _find_match
        if line_start is not None or line_end is not None:
            lines = content.splitlines(keepends=True)
            if not lines:
                return f"{RED}File is empty{RESET}"
            # Convert to int if string
            if isinstance(line_start, str):
                line_start = int(line_start)
            if isinstance(line_end, str):
                line_end = int(line_end)
            # Defaults
            if line_start is None:
                line_start = 1
            if line_end is None:
                line_end = line_start
            # Clamp to valid range
            line_start = max(1, min(line_start, len(lines)))
            line_end = max(line_start, min(line_end, len(lines)))
            # Build new content directly
            before = ''.join(lines[:line_start - 1])
            after = ''.join(lines[line_end:])
            # Ensure new_string ends with newline if replacing lines that had one
            if lines[line_end - 1].endswith('\n') and new_string and not new_string.endswith('\n'):
                new_string = new_string + '\n'
            new_content = before + new_string + after
            suffix = f"\n{DIM}(lines {line_start}-{line_end}){RESET}"
            return show_diff(content, new_content, p.name) + suffix

        if not old_string:
            return ""
        m = _find_match(content, old_string, new_string)

        if m['type'] == 'flexible':
            start, end, matched_text = m['match']
            adjusted_new = _adjust_replacement_indent(new_string, matched_text)
            new_content = content[:start] + adjusted_new + content[end:]
            return show_diff(content, new_content, p.name) + f"\n{DIM}(flexible match){RESET}"

        if m['type'] is not None:
            if replace_all:
                new_content = m['content'].replace(m['old'], m['new'])
            else:
                new_content = m['content'].replace(m['old'], m['new'], 1)
            suffix = f"\n{DIM}(first of {m['count']}){RESET}" if m['count'] > 1 and not replace_all else ""
            return show_diff(m['content'], new_content, p.name) + suffix
        else:
            # String not found - try to find close matches
            lines = content.splitlines()
            old_lines = old_string.splitlines()
            if old_lines:
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

def show_raw_feedback():
    """Display stored raw LLM feedback."""
    global _last_raw_response
    raw_content = _last_raw_response.get("content", "")
    reasoning = _last_raw_response.get("reasoning", "")
    tool_calls = _last_raw_response.get("tool_calls", [])

    if not raw_content and not reasoning and not tool_calls:
        print(f"{YELLOW}No raw response stored.{RESET}")
        return

    print(f"\n{DIM}{'─'*60}{RESET}")
    print(f"{CYAN}▼ Raw LLM Feedback{RESET}")
    print(f"{DIM}{'─'*60}{RESET}")

    if reasoning:
        print(f"{YELLOW}reasoning:{RESET}")
        for line in reasoning.split('\n'):
            print(f"  {DIM}{line}{RESET}")
        print()

    if raw_content:
        print(f"{YELLOW}content:{RESET}")
        for line in raw_content.split('\n'):
            print(f"  {DIM}{line}{RESET}")
        print()

    if tool_calls:
        print(f"{YELLOW}tool_calls:{RESET}")
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "unknown")
            args = fn.get("arguments", {})
            print(f"  {DIM}{name}({json.dumps(args, indent=4)}){RESET}")

    print(f"{DIM}{'─'*60}{RESET}")


def store_raw_response(raw_content: str, reasoning: str, tool_calls: list = None):
    """Store raw LLM response for later viewing with 'raw' command."""
    global _last_raw_response
    _last_raw_response = {
        "content": raw_content,
        "reasoning": reasoning,
        "tool_calls": tool_calls or []
    }


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
        return countdown_confirm(1)

    # Auto-confirm write tools with longer countdown (bash always requires manual approval)
    if name in ("edit_file", "write_file"):
        return countdown_confirm(10)

    try:
        confirm = input(f"{YELLOW}Execute?{RESET} [y/N]: ").strip().lower()
    except EOFError:
        confirm = "n"

    return confirm in ("y", "yes")

def show_plan_summary(tracker, max_todos: int = None) -> tuple:
    """Show plan summary if subtasks exist.

    Returns tuple: (plan_complete: bool, needs_consolidation: bool, preferred_count: int or None)
    """
    subtasks = [t for t in tracker.tasks if t.get("parent_id")]
    if not subtasks:
        return (False, False, None)

    # Check if too many todos
    if max_todos and len(subtasks) > max_todos:
        print(f"\n{YELLOW}⚠ {len(subtasks)} tasks created (max: {max_todos}){RESET}")
        for i, task in enumerate(subtasks, 1):
            location = ""
            if task.get("file_path") or task.get("anchor"):
                parts = []
                if task.get("file_path"):
                    parts.append(task["file_path"])
                if task.get("anchor"):
                    parts.append(task["anchor"])
                location = f" {DIM}@ {':'.join(parts)}{RESET}"
            print(f"  {i}. {task['content']}{location}")

        # Ask user for preferred count
        print(f"\n{CYAN}How many steps would you prefer?{RESET}")
        try:
            user_input = input(f"Enter a number (or press Enter to keep {len(subtasks)}): ").strip()
            if user_input.isdigit() and int(user_input) < len(subtasks):
                preferred = int(user_input)
                # Clear existing subtasks so LLM can recreate
                tracker.tasks = [t for t in tracker.tasks if not t.get("parent_id")]
                print(f"{DIM}Asking LLM to consolidate into {preferred} steps...{RESET}")
                return (False, True, preferred)
        except (EOFError, KeyboardInterrupt):
            pass
        # User accepted current count
        print(f"\n{GREEN}✓ Keeping {len(subtasks)} steps{RESET}")

    print(f"\n{GREEN}✓ Plan created with {len(subtasks)} steps:{RESET}")
    for i, task in enumerate(subtasks, 1):
        location = ""
        if task.get("file_path") or task.get("anchor"):
            parts = []
            if task.get("file_path"):
                parts.append(task["file_path"])
            if task.get("anchor"):
                parts.append(task["anchor"])
            location = f" {DIM}@ {':'.join(parts)}{RESET}"
        print(f"  {i}. {task['content']}{location}")
    print(f"\n{DIM}Type 'approve' to execute the plan.{RESET}")
    return (True, False, None)


def parse_validation_issues(issues_summary: str, modified_files: list[str] = None) -> list[dict]:
    """Parse validation issues summary into individual fix tasks.

    Args:
        issues_summary: The raw issues text from validation (e.g., "ISSUES FOUND:\n- **Line 53** – ...")
        modified_files: List of files that were modified (used to infer file_path)

    Returns:
        List of dicts with 'content', 'active_form', and optionally 'file_path', 'anchor'
    """
    issues = []

    # Infer file path from modified files (use first .py file if available)
    default_file = None
    if modified_files:
        for f in modified_files:
            if f.endswith('.py'):
                default_file = f
                break
        if not default_file:
            default_file = modified_files[0]

    # Match markdown list items: "- **Label** – description" or "- **Label**: description"
    # Handle both en-dash (–) and colon (:) as delimiters
    pattern = r'-\s*\*\*([^*]+)\*\*\s*[–:\-]\s*(.+?)(?=\n-\s*\*\*|\n\n|$)'
    matches = re.findall(pattern, issues_summary, re.DOTALL)

    for label, description in matches:
        label = label.strip()
        desc = description.strip()
        # Remove any trailing markdown or extra newlines
        desc = re.sub(r'\s+', ' ', desc)

        # Extract line numbers from label (e.g., "Line 53" or "Lines 94–99")
        line_match = re.search(r'[Ll]ines?\s*(\d+)', label)
        line_hint = f" (line {line_match.group(1)})" if line_match else ""

        # Extract function/method name from description (e.g., `Board.is_mine()`)
        anchor = None
        func_match = re.search(r'`(\w+\.\w+)\(`', desc)
        if func_match:
            anchor = func_match.group(1)

        # Truncate description for content
        if len(desc) > 120:
            desc = desc[:117] + "..."

        content = f"Fix{line_hint}: {desc}"
        active_form = f"Fixing issue at {label.lower()}"

        issue = {"content": content, "active_form": active_form}
        if default_file:
            issue["file_path"] = default_file
        if anchor:
            issue["anchor"] = anchor
        issues.append(issue)

    # If no structured issues found, try simpler parsing (plain list items)
    if not issues:
        lines = issues_summary.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') and len(line) > 3:
                desc = line[2:].strip()
                if desc and not desc.upper().startswith('ISSUES FOUND'):
                    if len(desc) > 150:
                        desc = desc[:147] + "..."
                    issue = {
                        "content": f"Fix: {desc}",
                        "active_form": "Fixing validation issue"
                    }
                    if default_file:
                        issue["file_path"] = default_file
                    issues.append(issue)

    return issues


def run_validation_stage(completed_task: dict, modified_files: list[str],
                         config: dict, tracker, messages: list) -> bool:
    """Run validation stage after task completion.

    Args:
        completed_task: The completed task dict
        modified_files: List of file paths modified during the task
        config: Configuration dict
        tracker: TaskTracker instance
        messages: Current conversation messages

    Returns:
        True if validation passed or user accepted, False if user wants to fix
    """
    if not modified_files:
        return True  # Nothing to validate

    print(f"\n{CYAN}◆ Validating task: {completed_task.get('content', 'unknown')[:50]}...{RESET}")

    # Build validation prompt
    system = VALIDATION_PROMPT.format(
        cwd=os.getcwd(),
        completed_task=completed_task.get('content', ''),
        modified_files="\n".join(f"- {f}" for f in modified_files)
    )

    # Create validation-specific messages
    val_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Validate the changes made for: {completed_task.get('content', '')}"}
    ]

    # Run validation with LLM
    client = OllamaClient(
        host=config["ollama"]["host"],
        timeout=config["ollama"]["timeout"]
    )

    # Use validation model if configured, otherwise use writing model
    validation_config = config.get("agent", {}).get("validation", {})
    model = validation_config.get("model") or config["models"].get("writing")

    # Save current mode
    previous_mode = tracker.mode
    tracker.mode = "validation"

    # Mini agent loop for validation (max 5 turns)
    issues_found = None
    issues_summary = ""
    timer = Timer()

    for _ in range(5):
        timer.start()
        try:
            response = client.chat_sync(
                model=model,
                messages=val_messages,
                tools=get_tool_schemas(),
                options={"temperature": 0.0, "num_ctx": 32768}
            )
        except Exception as e:
            timer.stop()
            print(f"{RED}✗ Validation error: {e}{RESET}")
            break
        timer.stop()
        elapsed = time.time() - timer.start_time

        content = response.get("message", {}).get("content", "")
        tool_calls = response.get("message", {}).get("tool_calls", [])

        # Check for issues in content
        if "ISSUES FOUND:" in content.upper():
            issues_found = True
            # Extract issues summary (everything after ISSUES FOUND:)
            idx = content.upper().find("ISSUES FOUND:")
            issues_summary = content[idx:].strip()
            break
        elif "NO ISSUES FOUND" in content.upper():
            issues_found = False
            break

        # Execute read-only tools
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                # Only allow read-only tools + bash (for linters/tests)
                print(f"\n{DIM}{'─'*60}{RESET}")
                print(f"{CYAN}◆{RESET} {DIM}[validation]{RESET} {name}")
                if args:
                    print(f"{DIM}{json.dumps(args, indent=2)}{RESET}")
                print(f"{DIM}{'─'*60}{RESET}")

                if name in ("read_file", "glob", "grep", "bash"):
                    result = execute_tool(name, args)
                    if result.startswith("Error"):
                        print(f"{RED}✗{RESET} {result[:500]}{'...' if len(result) > 500 else ''}")
                    else:
                        print(f"{GREEN}✓{RESET} {DIM}{result[:200]}{'...' if len(result) > 200 else ''}{RESET}")
                else:
                    result = f"Error: Tool '{name}' not available in validation mode"
                    print(f"{RED}✗{RESET} {result}")

                val_messages.append({"role": "assistant", "content": content, "tool_calls": [tc]})
                val_messages.append({"role": "tool", "content": result})
        else:
            # No tool calls and no issues determination - add to messages and continue
            val_messages.append({"role": "assistant", "content": content})
            val_messages.append({"role": "user", "content": "Please check the modified files and report: ISSUES FOUND: [list] or NO ISSUES FOUND"})

    # Restore mode
    tracker.mode = previous_mode

    if issues_found is True:
        print(f"\n{YELLOW}⚠ Validation found issues:{RESET}")
        print(f"{DIM}{issues_summary}{RESET}")

        try:
            response = input(f"\n{YELLOW}Fix issues?{RESET} [y/N/skip]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"

        if response in ("y", "yes"):
            # Parse issues and create subtasks
            parsed_issues = parse_validation_issues(issues_summary, modified_files)

            if parsed_issues:
                print(f"\n{CYAN}Creating {len(parsed_issues)} fix subtask(s):{RESET}")
                parent_id = completed_task.get("id")

                created_task_ids = []
                for issue in parsed_issues:
                    task_id = tracker.add(
                        content=issue["content"],
                        active_form=issue["active_form"],
                        parent_id=parent_id,
                        file_path=issue.get("file_path"),
                        anchor=issue.get("anchor")
                    )
                    created_task_ids.append(task_id)
                    file_hint = f" @ {issue.get('file_path', '')}" if issue.get('file_path') else ""
                    print(f"  {DIM}+ [{task_id}] {issue['content'][:50]}...{file_hint}{RESET}")

                # Add context message for the LLM with explicit instructions
                first_task_id = created_task_ids[0] if created_task_ids else "..."
                first_file = parsed_issues[0].get("file_path", "the file") if parsed_issues else "the file"
                messages.append({
                    "role": "user",
                    "content": (
                        f"VALIDATION ISSUES FOUND:\n{issues_summary}\n\n"
                        f"{len(parsed_issues)} FIX SUBTASKS ALREADY CREATED - DO NOT create new subtasks.\n\n"
                        f"Execute each fix subtask in order. For each one:\n"
                        f"1. todo(action='start', task_id='{first_task_id}') - mark in_progress\n"
                        f"2. read_file(path='{first_file}') - read the code first\n"
                        f"3. edit_file(path='...', old_string='<exact text from file>', new_string='<fixed code>')\n"
                        f"4. todo(action='complete', task_id='...') - mark done\n\n"
                        f"CRITICAL: You MUST read_file BEFORE edit_file. The old_string must be copied exactly from the file."
                    )
                })
            else:
                # Fallback if parsing fails - use generic fix request
                messages.append({
                    "role": "user",
                    "content": f"VALIDATION ISSUES FOUND:\n{issues_summary}\n\nPlease fix these issues before proceeding to the next task."
                })

            return False  # Signal to continue fixing
        else:
            print(f"{DIM}Skipping validation fixes.{RESET}")
    elif issues_found is False:
        print(f"{GREEN}✓ Validation passed{RESET}")
    else:
        print(f"{DIM}Validation inconclusive, continuing...{RESET}")

    return True


def agent_loop(task: str, config: dict, resume: bool = False) -> bool:
    """Main agent execution loop. Returns True if cancelled by ESC."""
    tracker = get_tracker()

    # Always reset on new tasks to prevent context leaking from previous sessions
    if not resume:
        reset_tracker()
        tracker = get_tracker()

    client = OllamaClient(
        host=config["ollama"]["host"],
        timeout=config["ollama"]["timeout"]
    )

    # Check Ollama is running
    if not client.is_available():
        print("Error: Ollama not available. Start it with: ollama serve")
        return False

    # Select model based on mode
    if "models" in config:
        if tracker.mode == "planning":
            model = config["models"].get("writing")  # prose generation
        elif tracker.mode == "todo":
            model = config["models"].get("planning")  # tool calling
        else:
            model = config["models"].get("writing")
    else:
        model = config.get("model")  # Backward compatibility

    available = client.list_models()
    if model not in available:
        print(f"Warning: Model '{model}' not found")
        print(f"Pull with: ollama pull {model}")
        print(f"Available: {', '.join(sorted(available))}")
        return False

    print(f"\n{MAGENTA}▶ errol{RESET} {DIM}using {model}{RESET}")
    if tracker.mode == "planning":
        print(f"{CYAN}◆ Planning mode{RESET} {DIM}(read-only, type 'approve' to execute){RESET}")

    # Extract reasoning level from task (ultrathink:, quick:, etc.)
    reasoning_level, clean_task = extract_reasoning_level(task)

    # Create root task if starting fresh
    root_id = None
    if not resume:
        root_id = tracker.add(clean_task, f"Working on: {clean_task[:50]}...")
        tracker.set_status(root_id, "in_progress")
        tracker.original_task = clean_task
        tracker.interrupted = False
    else:
        # Find root task ID from existing tasks
        for t in tracker.tasks:
            if not t.get("parent_id"):
                root_id = t["id"]
                break

    # Build system prompt based on mode
    if tracker.mode == "planning":
        system = ENRICHED_PLAN_PROMPT.format(cwd=os.getcwd())
    elif tracker.mode == "todo":
        system = TODO_EXTRACTION_PROMPT.format(
            enriched_plan=tracker.enriched_plan,
            root_id=root_id
        )
    else:
        system = SYSTEM_PROMPT.format(cwd=os.getcwd(), reasoning_level=reasoning_level)

    # Build messages based on mode
    # Todo mode always starts fresh - the enriched plan is in the system prompt
    if tracker.mode == "todo":
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Extract todos from the plan above. Call todo() for each change needed."}
        ]
    elif resume and tracker.messages:
        messages = tracker.messages
        # Update system prompt if mode changed (e.g., planning -> write)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system
        # In write mode, inject instruction to execute the plan (only once)
        if tracker.mode == "write" and tracker.tasks:
            last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            already_injected = last_user_msg and "Plan approved" in last_user_msg.get("content", "")
            if not already_injected:
                subtasks = [t for t in tracker.tasks if t.get("parent_id")]
                if subtasks:
                    task_list = "\n".join(f"- [{t['id']}] {t['content']} @ {t.get('file_path', '?')}" for t in subtasks)
                    messages.append({
                        "role": "user",
                        "content": f"Plan approved. Now implement these tasks in order:\n{task_list}\n\nStart with the FIRST task. Use write_file to create new files. After completing each task, call todo(action='complete', task_id='<id>') to mark it done. Do NOT create new tasks."
                    })
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": clean_task}
        ]

    # Select tools based on mode
    all_tools = get_tool_schemas()
    if tracker.mode == "planning":
        # Planning mode: read-only exploration tools only
        tools = [t for t in all_tools if t["function"]["name"] in ("glob", "grep", "read_file")]
    elif tracker.mode == "todo":
        # Todo mode: only the todo tool for extracting tasks
        tools = [t for t in all_tools if t["function"]["name"] == "todo"]
    else:
        # Write/validation modes: all tools (including todo for completing tasks)
        tools = all_tools
    allowed_tool_names = {t["function"]["name"] for t in tools}
    max_turns = config["agent"]["max_turns"]
    max_todos = config["agent"].get("max_todos")  # Optional limit on todo items
    timer = Timer()
    reprompt_count = 0
    max_reprompts = 5
    last_file_read = None  # Track last file for context in prompts
    tool_call_cache = {}  # Track glob/grep calls to prevent duplicates: cache_key -> result

    for turn in range(max_turns):
        # Get response (non-streaming for reliable tool calls)
        full_content = ""
        tool_calls = []

        timer.start()
        try:
            response = client.chat_sync(model, messages, tools=tools,
                                        options={"temperature": 0.0, "num_ctx": 32768})
        except KeyboardInterrupt:
            timer.stop()
            # Save state for continuation
            tracker.messages = messages
            tracker.interrupted = True
            tracker.save()
            print(f"\n{YELLOW}⚠ Interrupted - state saved. Type 'continue' to resume.{RESET}")
            return True  # Signal cancellation
        except Exception as e:
            # If Ollama fails to parse tool calls, retry (error may be transient)
            error_str = str(e)
            if "error parsing tool call" in error_str or "500" in error_str:
                print(f"{YELLOW}⚠ Ollama tool parsing failed, retrying...{RESET}")
                try:
                    # First retry with tools (transient errors)
                    response = client.chat_sync(model, messages, tools=tools,
                                                options={"temperature": 0.0, "num_ctx": 32768})
                except KeyboardInterrupt:
                    timer.stop()
                    tracker.messages = messages
                    tracker.interrupted = True
                    tracker.save()
                    print(f"\n{YELLOW}⚠ Interrupted - state saved. Type 'continue' to resume.{RESET}")
                    return True
                except Exception as e2:
                    # Both retries failed - inject guidance and continue to next iteration
                    timer.stop()
                    print(f"{YELLOW}⚠ Retry failed, prompting model to continue...{RESET}")
                    messages.append({
                        "role": "assistant",
                        "content": "(Tool call failed due to parsing error)"
                    })
                    messages.append({
                        "role": "user",
                        "content": "Tool parsing failed. Please continue with the current task. Use the tool functions (read_file, edit_file, write_file, todo, etc.) to make progress. Call the tools properly with valid JSON arguments."
                    })
                    continue  # Try again with the guidance prompt
            else:
                timer.stop()
                print(f"{RED}✗ API error: {e}{RESET}")
                break
        timer.stop()
        elapsed = time.time() - timer.start_time

        msg = response.get("message", {})
        full_content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        tool_calls = msg.get("tool_calls", [])

        # Filter out tool calls that aren't in the allowed tools for this mode
        # (LLM may generate calls for tools it wasn't given)
        ignored_tool_calls = []
        if tool_calls:
            filtered = []
            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "")
                if name in allowed_tool_names:
                    filtered.append(tc)
                else:
                    ignored_tool_calls.append(name)
                    print(f"{DIM}↻ ignoring disallowed tool call: {name}{RESET}")
            tool_calls = filtered

        # Fallback: parse tool calls from text if model doesn't use native format
        # Skip in planning mode - we expect text output, not tool calls
        if not tool_calls and full_content and tracker.mode != "planning":
            parsed = parse_tool_calls_from_text(full_content)
            # Also filter parsed tool calls
            tool_calls = [tc for tc in parsed if tc.get("function", {}).get("name", "") in allowed_tool_names]

        # Fallback: gpt-oss sometimes puts tool calls in reasoning_content
        if not tool_calls and reasoning and tracker.mode != "planning":
            parsed = parse_tool_calls_from_text(reasoning)
            tool_calls = [tc for tc in parsed if tc.get("function", {}).get("name", "") in allowed_tool_names]

        # Store raw response for 'raw' command (always, even if only tool calls)
        store_raw_response(full_content, reasoning, tool_calls)

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

        # In planning mode, text output IS the enriched plan
        if tracker.mode == "planning" and full_content.strip() and not tool_calls:
            tracker.enriched_plan = full_content
            tracker.write_enriched_plan(full_content)
            tracker.mode = "todo"  # Transition to next stage
            tracker.save()
            print(f"\n{DIM}Plan saved to .errol/PLAN.md{RESET}")
            print(f"{DIM}Type 'approve' to extract tasks.{RESET}")
            break

        # Add assistant message to history
        # Critical for gpt-oss: pass reasoning_content back to maintain chain-of-thought
        assistant_msg = {"role": "assistant", "content": full_content}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Execute tool calls with confirmation
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                original_name = fn.get("name", "")
                args = fn.get("arguments", {})

                # Skip invalid tool names (channel markers, garbage, etc.)
                if not original_name or any(c in original_name for c in '<>|'):
                    continue

                # Parse args if string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                # Resolve tool name (alias or fuzzy match) for consistent checks
                name, _ = resolve_tool_name(original_name)

                # Check for duplicate glob/grep/read_file calls
                if name in ("glob", "grep", "read_file"):
                    try:
                        cache_key = (name, json.dumps(args, sort_keys=True))
                    except:
                        cache_key = (name, str(args))
                    if cache_key in tool_call_cache:
                        print(f"\n{DIM}{'─'*60}{RESET}")
                        print(f"{CYAN}◆{RESET} {name}")
                        print(f"{DIM}{'─'*60}{RESET}")
                        print(f"{YELLOW}○ skipped (duplicate call){RESET}")
                        messages.append({"role": "tool", "content": tool_call_cache[cache_key]})
                        continue

                # Enforce planning mode - block write operations
                if tracker.mode == "planning" and name in ("edit_file", "write_file", "bash"):
                    print(f"\n{DIM}{'─'*60}{RESET}")
                    print(f"{CYAN}◆{RESET} {name}")
                    print(f"{DIM}{'─'*60}{RESET}")
                    print(f"{YELLOW}⚠ Planning mode - cannot modify files.{RESET}")
                    print(f"{DIM}Type 'approve' to switch to write mode.{RESET}")
                    result = "Error: Cannot modify files in planning mode. User must type 'approve' to switch to write mode."
                    messages.append({"role": "tool", "content": result})
                    continue

                # Capture task info before execution (task gets removed on complete)
                completing_task = None
                if name == "todo" and args.get("action") == "complete":
                    completing_task = tracker.get(args.get("task_id"))

                # Validate before prompting
                validation_error = validate_tool_call(name, args)
                if validation_error:
                    print(f"\n{DIM}{'─'*60}{RESET}")
                    print(f"{CYAN}◆{RESET} {name}")
                    print(f"{DIM}{'─'*60}{RESET}")
                    print(f"{RED}✗ Error: {validation_error}{RESET}")
                    result = f"Error: {validation_error}"
                elif confirm_tool(name, args):
                    result = execute_tool(name, args)
                    # Only treat as error if starts with "Error" (not if content contains it)
                    if result.startswith("Error"):
                        print(f"{RED}✗{RESET} {result[:500]}{'...' if len(result) > 500 else ''}")
                    else:
                        print(f"{GREEN}✓{RESET} {DIM}{result[:500]}{'...' if len(result) > 500 else ''}{RESET}")
                        # Track last successfully read file
                        if name == "read_file" and args.get("path"):
                            last_file_read = args["path"]
                        # Track file modifications for validation
                        if name in ("write_file", "edit_file") and args.get("path"):
                            tracker.add_modified_file(args["path"])
                            # Invalidate cached reads of this file so LLM re-reads fresh content
                            edited_path = str(Path(args["path"]).expanduser().resolve())
                            for key in list(tool_call_cache.keys()):
                                if key[0] == "read_file":
                                    try:
                                        cached_args = json.loads(key[1])
                                        cached_path = str(Path(cached_args.get("path", "")).expanduser().resolve())
                                        if cached_path == edited_path:
                                            del tool_call_cache[key]
                                    except:
                                        pass
                        # Cache glob/grep/read_file results for deduplication
                        if name in ("glob", "grep", "read_file"):
                            try:
                                cache_key = (name, json.dumps(args, sort_keys=True))
                            except:
                                cache_key = (name, str(args))
                            tool_call_cache[cache_key] = result
                            # In planning mode, prompt to read existing files before creating todos
                            if tracker.mode == "planning" and name == "glob" and result and not result.startswith("No"):
                                # Extract file paths from glob result
                                files_found = [f.strip() for f in result.strip().split('\n') if f.strip()]
                                if files_found:
                                    # Check if any todos already created - if so, don't re-prompt
                                    subtasks = [t for t in tracker.tasks if t.get("parent_id")]
                                    if not subtasks:
                                        messages.append({
                                            "role": "user",
                                            "content": f"Found existing files: {', '.join(files_found[:5])}. IMPORTANT: Read these files BEFORE creating todos to understand existing implementation. Use read_file on relevant files first."
                                        })
                        # Show next task after completing one, run validation only when all done
                        if name == "todo" and args.get("action") == "complete":
                            # Check for remaining subtasks (completed tasks are now removed)
                            remaining_subtasks = [t for t in tracker.tasks if t.get("parent_id")]
                            next_task = remaining_subtasks[0] if remaining_subtasks else None

                            if next_task:
                                # More tasks remain - show next task, don't validate yet
                                print(f"\n{DIM}{'─'*60}{RESET}")
                                print(f"{YELLOW}→{RESET} Task {next_task['id']}: {next_task['content']}")
                            else:
                                # All subtasks complete - run validation on all modified files
                                modified_files = tracker.get_modified_files()
                                validation_enabled = config.get("agent", {}).get("validation", {}).get("enabled", True)
                                if modified_files and validation_enabled and tracker.mode == "write":
                                    # Get root task for validation context
                                    root_task = None
                                    for t in tracker.tasks:
                                        if not t.get("parent_id"):
                                            root_task = t
                                            break
                                    validation_passed = run_validation_stage(
                                        root_task or {"content": "all tasks"},
                                        modified_files,
                                        config,
                                        tracker,
                                        messages
                                    )
                                    if not validation_passed:
                                        # User wants to fix - don't clear files, continue loop
                                        continue

                                # Clear modified files after validation
                                tracker.clear_modified_files()
                else:
                    result = "Tool execution skipped by user"
                    print(f"{YELLOW}○ skipped{RESET}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result
                })

                # Inject focused task reminder - tell LLM exactly what to do next
                # Include after todo(complete) to guide to next task
                if tracker.has_tasks() and tracker.mode == "write":
                    # Find the next pending or in-progress subtask
                    next_task = None
                    for t in tracker.tasks:
                        if t.get("parent_id") and t["status"] in ("pending", "in_progress"):
                            next_task = t
                            break
                    if next_task:
                        file_path = next_task.get("file_path")
                        file_exists = file_path and os.path.exists(file_path)

                        location = ""
                        if file_path:
                            location = f" in {file_path}"
                            if next_task.get("anchor"):
                                location += f" at {next_task['anchor']}"

                        # Infer task type from content to provide appropriate guidance
                        task_lower = next_task['content'].lower()
                        if any(kw in task_lower for kw in ('add ', 'create ', 'implement ', 'new ')):
                            task_guidance = "Add this as NEW code. Do NOT replace or delete existing methods/functions."
                        elif any(kw in task_lower for kw in ('remove ', 'delete ')):
                            task_guidance = "Remove the specified code."
                        elif any(kw in task_lower for kw in ('replace ', 'rewrite ')):
                            task_guidance = "Replace the existing implementation entirely."
                        elif any(kw in task_lower for kw in ('modify ', 'update ', 'change ', 'edit ', 'fix ')):
                            task_guidance = "Modify the existing code. Keep the structure but update the implementation."
                        else:
                            task_guidance = "Make the required changes."

                        if file_exists:
                            action = f"Use edit_file to modify this file. {task_guidance}"
                        elif file_path:
                            action = f"Use write_file to CREATE {file_path}."
                        else:
                            action = f"Use the appropriate tool. {task_guidance}"

                        # Include task-specific specification if available
                        spec = next_task.get("specification", "")
                        spec_context = f"\n\nSpecification:\n{spec}" if spec else ""

                        messages.append({
                            "role": "user",
                            "content": f"NEXT TASK: {next_task['content']}{location}{spec_context}\n\n{action} When done, call todo(action='complete', task_id='{next_task['id']}')."
                        })

                if name == "read_file" and not result.startswith("Error") and tracker.mode == "planning":
                    messages.append({
                        "role": "user",
                        "content": "Now create todos for each change needed. Use todo(action='add', ...) for each step."
                    })

            # After processing all tool calls, check if todos are complete
            if tracker.mode == "todo":
                plan_complete, needs_consolidation, preferred_count = show_plan_summary(tracker, max_todos)
                if needs_consolidation:
                    messages.append({
                        "role": "user",
                        "content": f"The plan has too many steps. Please consolidate into exactly {preferred_count} high-level tasks. Call todo(action='add', ...) for each consolidated step."
                    })
                elif plan_complete:
                    print(f"\n{DIM}Type 'approve' to execute these tasks.{RESET}")
                    break
        else:
            # In planning mode, if we ignored tool calls (like todo), reprompt for text plan
            if tracker.mode == "planning" and ignored_tool_calls and reprompt_count < max_reprompts:
                reprompt_count += 1
                print(f"{DIM}↻ prompting to output plan as text ({reprompt_count}/{max_reprompts})...{RESET}")
                messages.append({
                    "role": "user",
                    "content": "In planning mode, you cannot use the todo tool. Instead, output a detailed implementation plan as markdown text. Include file paths, function signatures, and integration points. Start with '# Implementation Plan:' and describe each change needed."
                })
                continue

            # No tool calls - check if all tasks are done (completed tasks are removed)
            # Skip this check in planning/todo modes where we're still building the plan
            remaining_subtasks = [t for t in tracker.tasks if t.get("parent_id")]
            if not remaining_subtasks and tracker.has_tasks() and tracker.mode not in ("planning", "todo"):
                # All planned tasks completed - allow model to finish
                break

            # Check if model described changes without using tools
            if looks_like_code_suggestion(full_content) and reprompt_count < max_reprompts:
                reprompt_count += 1

                # In todo mode: force todo creation
                if tracker.mode == "todo":
                    plan_complete, needs_consolidation, preferred_count = show_plan_summary(tracker, max_todos)
                    if needs_consolidation:
                        messages.append({
                            "role": "user",
                            "content": f"The plan has too many steps. Please consolidate into exactly {preferred_count} high-level tasks. Call todo(action='add', ...) for each consolidated step."
                        })
                        continue
                    elif plan_complete:
                        break
                    # No subtasks yet - prompt to create them
                    print(f"{DIM}↻ prompting to create todos (planning mode) ({reprompt_count}/{max_reprompts})...{RESET}")
                    messages.append({
                        "role": "user",
                        "content": "You described changes but did NOT create todos. In planning mode you MUST call todo(action='add', content='...', file_path='...', anchor='...') for EACH step. Do NOT describe the plan in text - call the todo tool now for each step."
                    })
                # In write mode: prompt for edits or todo tracking
                elif looks_like_multi_step_plan(full_content) and not tracker.has_tasks():
                    print(f"{DIM}↻ prompting to create todos for multi-step plan ({reprompt_count}/{max_reprompts})...{RESET}")
                    messages.append({
                        "role": "user",
                        "content": "You outlined multiple changes but did not track them. Before making edits, call the todo tool with action='add' for EACH step in your plan. This ensures you complete ALL changes, not just the first one. Create the todos now."
                    })
                else:
                    print(f"{DIM}↻ prompting to use edit_file ({reprompt_count}/{max_reprompts})...{RESET}")
                    file_hint = f" The file you read was: {last_file_read}" if last_file_read else ""
                    messages.append({
                        "role": "user",
                        "content": f"You described a code change but did not call edit_file. Describing changes does NOT modify the file. You MUST call edit_file with path, old_string (exact text to replace), and new_string (replacement).{file_hint} Do it now."
                    })
            # Check if model is asking a question
            elif looks_like_question(full_content) and reprompt_count < max_reprompts:
                reprompt_count += 1
                print(f"{DIM}↻ continuing autonomously ({reprompt_count}/{max_reprompts})...{RESET}")
                messages.append({
                    "role": "user",
                    "content": "Continue working on the task autonomously. Don't ask for input - make reasonable choices and proceed with the next step."
                })
            else:
                # Truly done or max reprompts reached
                break

    if turn >= max_turns - 1:
        print(f"\n{YELLOW}⚠ Reached max turns ({max_turns}){RESET}")
        # Save state so user can continue
        tracker.messages = messages
        tracker.interrupted = True
        tracker.save()
        print(f"{DIM}State saved. Type 'continue' to resume.{RESET}")
    else:
        # Check if all subtasks are completed (completed tasks are removed)
        remaining_subtasks = [t for t in tracker.tasks if t.get("parent_id")]
        if remaining_subtasks:
            # Still have pending work - save state for continuation
            print(f"\n{YELLOW}⚠ {len(remaining_subtasks)} tasks remaining:{RESET}")
            for t in remaining_subtasks[:3]:
                print(f"  {DIM}- {t['content']}{RESET}")
            if len(remaining_subtasks) > 3:
                print(f"  {DIM}... and {len(remaining_subtasks) - 3} more{RESET}")
            tracker.messages = messages
            tracker.interrupted = True
            tracker.save()
            print(f"{DIM}State saved. Type 'continue' to resume.{RESET}")
        else:
            # All tasks completed - remove root task and clear session
            for task in tracker.list():
                if task.get("parent_id") is None:
                    tracker.remove(task["id"])
            clear_saved_session()

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
        print(f"{RED}✗ Ollama not available{RESET}")
        return

    print(f"\n{MAGENTA}▶ errol{RESET} {DIM}models{RESET}")

    print(f"\n{CYAN}◆{RESET} Available")
    for m in client.list_models():
        print(f"  {DIM}{m}{RESET}")

    if "models" in config:
        print(f"\n{CYAN}◆{RESET} Configured")
        planning = config['models'].get('planning', 'not set')
        writing = config['models'].get('writing', 'not set')
        print(f"  {DIM}planning:{RESET} {planning}")
        print(f"  {DIM}writing:{RESET}  {writing}")
    else:
        configured = config.get('model', 'not set')
        print(f"\n{CYAN}◆{RESET} Configured")
        print(f"  {DIM}model:{RESET} {configured}")


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
    print(f"\n{CYAN}◆{RESET} {BOLD}Unit Tests (tools){RESET}")
    from test_tools import run_all_tests as run_tool_tests
    results = run_tool_tests()

    print(f"\n{CYAN}◆{RESET} {BOLD}Unit Tests (task_tracker){RESET}")
    from test_task_tracker import run_all_tests as run_tracker_tests
    tracker_results = run_tracker_tests()

    print(f"\n{CYAN}◆{RESET} {BOLD}Unit Tests (validation){RESET}")
    from test_validation import run_all_tests as run_validation_tests
    validation_results = run_validation_tests()

    print(f"\n{CYAN}◆{RESET} {BOLD}Unit Tests (cache){RESET}")
    from test_cache import run_all_tests as run_cache_tests
    cache_results = run_cache_tests()

    # Combine results
    total_passed = results.passed + tracker_results.passed + validation_results.passed + cache_results.passed
    total_failed = results.failed + tracker_results.failed + validation_results.failed + cache_results.failed
    all_errors = results.errors + tracker_results.errors + validation_results.errors + cache_results.errors

    if total_failed > 0:
        print(f"\n{RED}✗{RESET} {total_passed} passed, {total_failed} failed")
        print(f"\n{DIM}Failures:{RESET}")
        for name, msg in all_errors:
            print(f"  {RED}✗{RESET} {name}: {DIM}{msg}{RESET}")
        sys.exit(1)

    print(f"\n{GREEN}✓{RESET} All {total_passed} tests passed")


if __name__ == "__main__":
    # Set program name for help output
    sys.argv[0] = "errol"
    app()
