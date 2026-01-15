"""Tool registry and implementations."""
import os
import glob as globlib
import subprocess
import difflib
from pathlib import Path
from typing import Optional

from task_tracker import get_tracker

# ANSI colors for diff output
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Parameter aliases - map common alternative param names to canonical names
# Format: {tool_name: {alias_param: canonical_param}}
PARAM_ALIASES = {
    "grep": {"query": "pattern", "search": "pattern", "regex": "pattern", "text": "pattern"},
    "glob": {"query": "pattern", "search": "pattern", "name": "pattern"},
    "read_file": {"file": "path", "filename": "path", "file_path": "path"},
    "write_file": {"file": "path", "filename": "path", "file_path": "path"},
    "edit_file": {"file": "path", "filename": "path", "file_path": "path"},
}

# Tool aliases - map common alternative names to canonical tool names
TOOL_ALIASES = {
    # Common variations
    "search": "grep",
    "find": "glob",
    "find_files": "glob",
    "search_files": "grep",
    "cat": "read_file",
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
    "run": "bash",
    "execute": "bash",
    "shell": "bash",
    "exec": "bash",
    "rg": "grep",
    "ripgrep": "grep",
    # Hallucinated names seen in practice
    "file_read": "read_file",
    "file_write": "write_file",
    "file_edit": "edit_file",
    "run_command": "bash",
    "search_content": "grep",
    # Todo tool aliases
    "task": "todo",
    "tasks": "todo",
    "todo_list": "todo",
    "track": "todo",
}


def show_diff(old: str, new: str, path: str) -> str:
    """Generate a colored unified diff."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}")

    result = []
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            result.append(f"{GREEN}{line.rstrip()}{RESET}")
        elif line.startswith('-') and not line.startswith('---'):
            result.append(f"{RED}{line.rstrip()}{RESET}")
        elif line.startswith('@@'):
            result.append(f"{CYAN}{line.rstrip()}{RESET}")
        else:
            result.append(line.rstrip())

    return "\n".join(result)

def read_file(path: str = None, offset: int = None, limit: int = None) -> str:
    """Read a file with optional line offset and limit."""
    try:
        if not path:
            return "Error: 'path' parameter is required"
        # Strip whitespace - models sometimes add leading/trailing spaces
        path = path.strip()
        # Handle None, "null" string, and type conversions from JSON
        if offset is None or offset == "null":
            offset = 0
        else:
            offset = int(offset)
        if limit is None or limit == "null":
            limit = 2000
        else:
            limit = int(limit)

        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        lines = p.read_text().splitlines()
        selected = lines[offset:offset + limit]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str = None, content: str = None) -> str:
    """Write content to a file."""
    try:
        if not path:
            return "Error: 'path' parameter is required"
        path = path.strip()
        if content is None:
            return "Error: 'content' parameter is required (the text to write to the file)"
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def _clean_string(s: str) -> str:
    """Clean string by removing line number prefixes, trailing whitespace, and normalizing quotes."""
    import re
    # Normalize smart quotes and special characters
    replacements = {
        '\u2018': "'", '\u2019': "'",  # curly single quotes
        '\u201c': '"', '\u201d': '"',  # curly double quotes
        '\u00a0': ' ',                  # non-breaking space
        '\u2013': '-', '\u2014': '-',  # en/em dashes
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    lines = s.split('\n')
    cleaned = []
    for line in lines:
        # Remove line number prefix (e.g., "123\t" or "  45\t")
        line = re.sub(r'^\s*\d+\t', '', line)
        # Strip trailing whitespace
        line = line.rstrip()
        cleaned.append(line)
    return '\n'.join(cleaned)

def _find_block_flexible(content: str, search: str) -> Optional[tuple]:
    """Find search block by matching line content, ignoring leading whitespace.

    Returns (start_pos, end_pos, matched_text) or None if not found.
    """
    content_lines = content.split('\n')
    search_lines = _clean_string(search).split('\n')

    # Strip leading/trailing whitespace from search lines for matching
    search_stripped = [line.strip() for line in search_lines if line.strip()]

    if not search_stripped:
        return None

    for i in range(len(content_lines) - len(search_stripped) + 1):
        match = True
        for j, search_line in enumerate(search_stripped):
            content_stripped = content_lines[i + j].strip()
            if content_stripped != search_line:
                match = False
                break

        if match:
            # Calculate positions in original content
            start_pos = sum(len(l) + 1 for l in content_lines[:i])
            matched_lines = content_lines[i:i + len(search_stripped)]
            matched_text = '\n'.join(matched_lines)
            end_pos = start_pos + len(matched_text)
            return start_pos, end_pos, matched_text

    return None

def _find_match(content: str, old_string: str, new_string: str = None):
    """Find old_string in content using fallback matching strategies.

    Returns dict with:
        - 'type': 'exact', 'cleaned', 'flexible', or None if not found
        - 'count': number of matches (for exact/cleaned)
        - 'old': actual old_string to use for replacement
        - 'new': actual new_string to use for replacement
        - 'content': content to use (may be normalized)
        - 'match': (start, end, matched_text) for flexible match
    """
    if new_string is None:
        new_string = ""

    # Try exact match
    count = content.count(old_string)
    if count > 0:
        return {
            'type': 'exact',
            'count': count,
            'old': old_string,
            'new': new_string,
            'content': content,
            'match': None
        }

    # Fallback 1: cleaned strings (line numbers, trailing whitespace)
    cleaned_old = _clean_string(old_string)
    cleaned_new = _clean_string(new_string)
    content_normalized = '\n'.join(line.rstrip() for line in content.split('\n'))
    count = content_normalized.count(cleaned_old)
    if count > 0:
        return {
            'type': 'cleaned',
            'count': count,
            'old': cleaned_old,
            'new': cleaned_new,
            'content': content_normalized,
            'match': None
        }

    # Fallback 2: flexible line-based matching
    match = _find_block_flexible(content, old_string)
    if match:
        return {
            'type': 'flexible',
            'count': 1,
            'old': old_string,
            'new': new_string,
            'content': content,
            'match': match
        }

    return {'type': None, 'count': 0, 'old': old_string, 'new': new_string, 'content': content, 'match': None}

def _adjust_replacement_indent(new_string: str, original_match: str) -> str:
    """Adjust new_string to use the same indentation style as original_match."""
    orig_lines = original_match.split('\n')
    new_lines = _clean_string(new_string).split('\n')

    # Detect indent unit from original (first indented content after base)
    orig_base_indent = ''
    orig_indent_unit = '\t'  # default to tab
    for line in orig_lines:
        if line.strip():
            orig_base_indent = line[:len(line) - len(line.lstrip())]
            # Detect if using spaces or tabs
            if orig_base_indent and orig_base_indent[0] == ' ':
                orig_indent_unit = orig_base_indent  # use full indent as unit
            elif orig_base_indent:
                orig_indent_unit = '\t'
            break

    # Detect indent unit from new_string
    new_base_indent = ''
    new_indent_unit = '    '  # default to 4 spaces
    for line in new_lines:
        if line.strip():
            new_base_indent = line[:len(line) - len(line.lstrip())]
            if new_base_indent and new_base_indent[0] == '\t':
                new_indent_unit = '\t'
            elif new_base_indent:
                new_indent_unit = new_base_indent
            break

    # Calculate base indent level
    def count_indent_level(indent: str, unit: str) -> int:
        if not unit or not indent:
            return 0
        if unit == '\t':
            return indent.count('\t')
        return len(indent) // len(unit) if unit else 0

    orig_base_level = count_indent_level(orig_base_indent, orig_indent_unit)
    new_base_level = count_indent_level(new_base_indent, new_indent_unit)

    # Build result with adjusted indentation
    result = []
    for i, line in enumerate(new_lines):
        if not line.strip():
            result.append('')
            continue

        line_indent = line[:len(line) - len(line.lstrip())]
        line_level = count_indent_level(line_indent, new_indent_unit)

        # Calculate relative level from base
        relative_level = line_level - new_base_level

        # Apply original's base level plus relative
        target_level = orig_base_level + relative_level
        new_indent = orig_indent_unit * target_level
        result.append(new_indent + line.lstrip())

    return '\n'.join(result)

def edit_file(path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False) -> str:
    """Replace old_string with new_string in file."""
    try:
        if not path:
            return "Error: 'path' parameter is required"
        path = path.strip()
        if old_string is None:
            return "Error: 'old_string' parameter is required (the text to find and replace)"
        if new_string is None:
            return "Error: 'new_string' parameter is required (the replacement text)"
        if old_string == new_string:
            return "Error: old_string and new_string are identical - no change needed"
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text()
        m = _find_match(content, old_string, new_string)

        if m['type'] == 'flexible':
            start, end, matched_text = m['match']
            adjusted_new = _adjust_replacement_indent(new_string, matched_text)
            new_content = content[:start] + adjusted_new + content[end:]
            p.write_text(new_content)
            return f"Edited {path}: applied changes (flexible match)"

        if m['type'] is None:
            # Show first line and try to find similar content
            first_line = old_string.split('\n')[0]
            search_stripped = first_line.strip()
            for i, file_line in enumerate(content.splitlines(), 1):
                if search_stripped and search_stripped in file_line:
                    return f"Error: String not found in {path}.\nLooking for: {first_line[:60]!r}\nSimilar at line {i}: {file_line[:60]!r}\nCheck indentation (tabs vs spaces)."
                # Check for near-matches with invisible character differences
                file_stripped = file_line.strip()
                if file_stripped and len(file_stripped) == len(search_stripped):
                    diffs = sum(1 for a, b in zip(file_stripped, search_stripped) if a != b)
                    if 0 < diffs <= 3:
                        search_hex = ' '.join(f'{ord(c):02x}' for c in search_stripped[:30])
                        file_hex = ' '.join(f'{ord(c):02x}' for c in file_stripped[:30])
                        return f"Error: String not found in {path}.\nLooking for: {search_stripped[:50]!r}\nNear match at line {i}: {file_stripped[:50]!r}\nSearch hex: {search_hex}\nFile hex:   {file_hex}"
            return f"Error: String not found in {path}. Looking for: {first_line[:80]!r}"

        if replace_all:
            new_content = m['content'].replace(m['old'], m['new'])
            replaced_msg = f" ({m['count']} occurrences)" if m['count'] > 1 else ""
        else:
            new_content = m['content'].replace(m['old'], m['new'], 1)
            replaced_msg = f" (first of {m['count']})" if m['count'] > 1 else ""

        p.write_text(new_content)
        return f"Edited {path}: applied changes{replaced_msg}"
    except Exception as e:
        return f"Error editing file: {e}"

def run_bash(command: str = None, timeout: int = 120, cwd: str = None, cmd: str = None) -> str:
    """Execute a bash command."""
    try:
        # Handle 'cmd' as alias for 'command'
        command = command or cmd
        if not command:
            return "Error: No command provided"
        # Handle cmd passed as array (e.g. ["bash", "-c", "..."])
        if isinstance(command, list):
            # If it's ["bash", "-c", "actual command"] or ["bash", "-lc", "..."], extract the command
            if len(command) >= 3 and command[0] == "bash" and command[1] in ("-c", "-lc"):
                command = command[2]
            else:
                command = " ".join(str(c) for c in command)
        # Handle string timeout from JSON
        if isinstance(timeout, str):
            timeout = int(timeout)
        work_dir = cwd if cwd else os.getcwd()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir
        )
        output = result.stdout + result.stderr
        if len(output) > 10000:
            output = output[:10000] + "\n... (truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"

# Directories to always exclude from glob results
GLOB_EXCLUDES = {'venv', 'node_modules', '__pycache__', '.git', '.venv', 'env', '.env',
                 'dist', 'build', '.tox', '.pytest_cache', '.mypy_cache', 'eggs', '*.egg-info',
                 '.errol'}

def glob_files(pattern: str = None, path: str = ".") -> str:
    """Find files matching a glob pattern."""
    try:
        if not pattern:
            return "Error: 'pattern' parameter is required"
        # Strip whitespace - models sometimes add leading/trailing spaces
        path = path.strip() if path else "."
        pattern = pattern.strip() if pattern else pattern
        base = Path(path).expanduser().resolve()
        matches = list(base.glob(pattern))

        # Filter out excluded directories
        def should_include(p: Path) -> bool:
            parts = p.relative_to(base).parts
            return not any(part in GLOB_EXCLUDES for part in parts)

        matches = [m for m in matches if should_include(m)]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not matches:
            return "No files matched"

        results = [str(m.relative_to(base)) for m in matches[:100]]
        return "\n".join(results)
    except Exception as e:
        return f"Error globbing: {e}"

def grep(pattern: str = None, path: str = ".", include: str = None) -> str:
    """Search for text pattern in files."""
    try:
        if not pattern:
            return "Error: 'pattern' parameter is required"
        import re
        path = path.strip() if path else "."
        base = Path(path).expanduser().resolve()

        # Determine files to search
        if base.is_file():
            files = [base]
        else:
            # Search all text files, respecting excludes
            glob_pattern = include if include else "**/*"
            files = [f for f in base.glob(glob_pattern) if f.is_file()]
            files = [f for f in files if not any(part in GLOB_EXCLUDES for part in f.relative_to(base).parts)]

        results = []
        try:
            regex = re.compile(pattern)
        except re.error:
            # Treat as literal string if not valid regex
            regex = re.compile(re.escape(pattern))

        for f in files[:500]:  # Limit files searched
            try:
                content = f.read_text(errors='ignore')
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel_path = f.relative_to(base) if base.is_dir() else f.name
                        results.append(f"{rel_path}:{i}: {line[:200]}")
                        if len(results) >= 50:  # Limit results
                            results.append("... (truncated)")
                            return "\n".join(results)
            except:
                continue

        if not results:
            return f"No matches for '{pattern}'"
        return "\n".join(results)
    except Exception as e:
        return f"Error searching: {e}"


def todo_tool(action: str = None, content: str = None, active_form: str = None,
              task_id: str = None, parent_id: str = None,
              file_path: str = None, anchor: str = None) -> str:
    """Manage task tracking for multi-step work.

    Actions:
    - list: Show all tasks with status
    - add: Add a new task (requires content, optionally active_form, parent_id, file_path, anchor)
    - start: Mark task as in_progress (requires task_id)
    - complete: Mark task as completed (requires task_id)

    For 'add' action:
    - file_path: Optional file path this task affects
    - anchor: Optional semantic anchor (function/class name) for locating the edit
    """
    tracker = get_tracker()

    if not action:
        return "Error: 'action' parameter is required. Use: list, add, start, complete"

    action = action.lower().strip()

    if action == "list":
        if not tracker.has_tasks():
            return "No tasks tracked yet."
        return tracker.to_prompt_context()

    elif action == "add":
        if not content:
            return "Error: 'content' parameter is required for 'add' action"
        new_task_id = tracker.add(content, active_form, parent_id, file_path, anchor)
        # Build response with location info if provided
        location = ""
        if file_path or anchor:
            parts = [p for p in [file_path, anchor] if p]
            location = f" @ {':'.join(parts)}"
        if parent_id:
            return f"Added subtask '{content}'{location} with ID: {new_task_id} (parent: {parent_id})"
        return f"Added task '{content}'{location} with ID: {new_task_id}"

    elif action == "start":
        if not task_id:
            return "Error: 'task_id' parameter is required for 'start' action"
        if tracker.set_status(task_id, "in_progress"):
            return f"Task {task_id} marked as in_progress"
        return f"Error: Task '{task_id}' not found"

    elif action == "complete":
        if not task_id:
            return "Error: 'task_id' parameter is required for 'complete' action"
        if tracker.set_status(task_id, "completed"):
            return f"Task {task_id} marked as completed"
        return f"Error: Task '{task_id}' not found"

    else:
        return f"Error: Unknown action '{action}'. Use: list, add, start, complete"


# Tool registry with schemas for Ollama
TOOLS = {
    "read_file": {
        "fn": read_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "offset": {"type": "integer", "description": "Line offset (default 0)"},
                        "limit": {"type": "integer", "description": "Max lines (default 2000)"}
                    },
                    "required": ["path"]
                }
            }
        }
    },
    "write_file": {
        "fn": write_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. WARNING: This will overwrite existing files. Always read_file first to check if the file exists and see its contents before overwriting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    },
    "edit_file": {
        "fn": edit_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace a string in a file with new content. String must be unique unless replace_all=true.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "old_string": {"type": "string", "description": "String to find"},
                        "new_string": {"type": "string", "description": "Replacement string"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false, requires unique match)"}
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        }
    },
    "bash": {
        "fn": run_bash,
        "schema": {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"}
                    },
                    "required": ["command"]
                }
            }
        }
    },
    "glob": {
        "fn": glob_files,
        "schema": {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by filename pattern (e.g. **/*.py). Does NOT search file contents - use grep for that.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.py)"},
                        "path": {"type": "string", "description": "Base directory (default .)"}
                    },
                    "required": ["pattern"]
                }
            }
        }
    },
    "grep": {
        "fn": grep,
        "schema": {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for text pattern inside files. Returns matching lines with file:line: prefix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                        "path": {"type": "string", "description": "File or directory to search (default .)"},
                        "include": {"type": "string", "description": "Glob pattern to filter files (e.g. **/*.py)"}
                    },
                    "required": ["pattern"]
                }
            }
        }
    },
    "todo": {
        "fn": todo_tool,
        "schema": {
            "type": "function",
            "function": {
                "name": "todo",
                "description": "Manage task tracking. Use this to track progress on multi-step tasks. Actions: list (show tasks), add (create task with optional parent_id for subtasks, file_path and anchor for location hints), start (mark in_progress), complete (mark done).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "Action: list, add, start, complete"},
                        "content": {"type": "string", "description": "Task description (for 'add' action)"},
                        "active_form": {"type": "string", "description": "Present tense form (for 'add' action, e.g. 'Reading file')"},
                        "task_id": {"type": "string", "description": "Task ID (for 'start' and 'complete' actions)"},
                        "parent_id": {"type": "string", "description": "Parent task ID to create a subtask (for 'add' action)"},
                        "file_path": {"type": "string", "description": "File path this task affects (for 'add' action)"},
                        "anchor": {"type": "string", "description": "Semantic anchor like function/class name for locating the edit (for 'add' action)"}
                    },
                    "required": ["action"]
                }
            }
        }
    }
}


def resolve_tool_name(name: str) -> tuple[str, bool]:
    """Resolve tool name, with alias and fuzzy matching fallback.

    Returns (resolved_name, was_fuzzy_match).
    """
    from difflib import get_close_matches

    # Exact match
    if name in TOOLS:
        return name, False

    # Check aliases
    if name in TOOL_ALIASES:
        return TOOL_ALIASES[name], False

    # Try fuzzy match against both tool names and aliases
    all_names = list(TOOLS.keys()) + list(TOOL_ALIASES.keys())
    matches = get_close_matches(name, all_names, n=1, cutoff=0.6)
    if matches:
        matched = matches[0]
        # If matched an alias, resolve to canonical name
        resolved = TOOL_ALIASES.get(matched, matched)
        print(f"{YELLOW}⚠ Fuzzy matched '{name}' → '{resolved}'{RESET}")
        return resolved, True

    return name, False  # Return original, will fail validation


def validate_tool_call(name: str, args: dict) -> Optional[str]:
    """Pre-validate a tool call. Returns error string if invalid, None if OK."""
    import re

    # Resolve tool name (alias or fuzzy match)
    resolved_name, _ = resolve_tool_name(name)

    # Check tool exists
    if resolved_name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool '{name}'. Available tools: {available}"

    # Check required parameters
    tool_info = TOOLS.get(resolved_name, {})
    schema = tool_info.get("schema", {}).get("function", {}).get("parameters", {})
    required = schema.get("required", [])
    for param in required:
        if param not in args or args.get(param) in (None, ""):
            return f"'{param}' parameter is required"

    # Check for placeholders
    for key, val in args.items():
        if isinstance(val, str) and re.search(r'<[a-zA-Z_-]+>', val):
            return f"Placeholder detected in '{key}': {val}. Use actual values."

    # Validate edit_file: check string exists
    if resolved_name == "edit_file":
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        if path and old_string:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"File not found: {path}"
            try:
                content = p.read_text()
                m = _find_match(content, old_string)
                if m['type'] is None:
                    first_line = old_string.split('\n')[0][:80]
                    return f"String not found in {path}. Looking for: {first_line!r}"
            except Exception as e:
                return f"Cannot read file: {e}"

    # Validate read_file: check file exists
    if resolved_name == "read_file":
        path = args.get("path", "")
        if path:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"File not found: {path}"
            if not p.is_file():
                return f"Not a file: {path}"

    return None  # Valid


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name with arguments."""
    import inspect
    import re

    # Early rejection of obviously invalid tool names (channel markers, etc.)
    if not name or any(c in name for c in '<>|'):
        return f"Error: Invalid tool name '{name}'"

    # Resolve tool name (alias or fuzzy match)
    resolved_name, _ = resolve_tool_name(name)

    if resolved_name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"Error: Unknown tool '{name}'. Available tools: {available}"

    # Reject placeholder values like <filename>, <path>, etc.
    for key, val in args.items():
        if isinstance(val, str) and re.search(r'<[a-zA-Z_-]+>', val):
            return f"Error: Placeholder detected in '{key}': {val}. Use actual values, not placeholders."

    # Remap parameter aliases (e.g., query -> pattern for grep)
    if resolved_name in PARAM_ALIASES:
        param_map = PARAM_ALIASES[resolved_name]
        remapped_args = {}
        for key, val in args.items():
            canonical_key = param_map.get(key, key)
            # Don't overwrite if canonical param already exists
            if canonical_key not in remapped_args:
                remapped_args[canonical_key] = val
        args = remapped_args

    # Auto-convert relative paths to absolute for path arguments
    if 'path' in args and args['path']:
        p = args['path'].strip()
        if not p.startswith('/') and not p.startswith('~'):
            args['path'] = str(Path.cwd() / p)

    fn = TOOLS[resolved_name]["fn"]
    # Filter args to only include valid parameters for the function
    sig = inspect.signature(fn)
    valid_params = set(sig.parameters.keys())
    filtered_args = {k: v for k, v in args.items() if k in valid_params}
    return fn(**filtered_args)

def get_tool_schemas() -> list[dict]:
    """Get all tool schemas for Ollama."""
    return [t["schema"] for t in TOOLS.values()]
