"""Tool registry and implementations."""
import os
import glob as globlib
import subprocess
import difflib
from pathlib import Path
from typing import Optional

# ANSI colors for diff output
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

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
        count = content.count(old_string)

        if count == 0:
            return f"Error: String not found in file"
        if count > 1 and not replace_all:
            return f"Error: String found {count} times, must be unique. Use replace_all=true to replace all occurrences."

        new_content = content.replace(old_string, new_string)
        p.write_text(new_content)
        replaced_msg = f" ({count} occurrences)" if count > 1 else ""
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
                 'dist', 'build', '.tox', '.pytest_cache', '.mypy_cache', 'eggs', '*.egg-info'}

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
                "description": "Find files matching a pattern",
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
    }
}

def validate_tool_call(name: str, args: dict) -> Optional[str]:
    """Pre-validate a tool call. Returns error string if invalid, None if OK."""
    import re

    # Check tool exists
    if name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool '{name}'. Available tools: {available}"

    # Check for placeholders
    for key, val in args.items():
        if isinstance(val, str) and re.search(r'<[a-zA-Z_-]+>', val):
            return f"Placeholder detected in '{key}': {val}. Use actual values."

    # Validate edit_file: check string exists and is unique (unless replace_all)
    if name == "edit_file":
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        replace_all = args.get("replace_all", False)
        if path and old_string:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"File not found: {path}"
            try:
                content = p.read_text()
                count = content.count(old_string)
                if count == 0:
                    return "String not found in file"
                if count > 1 and not replace_all:
                    return f"String found {count} times, must be unique. Use replace_all=true to replace all."
            except Exception as e:
                return f"Cannot read file: {e}"

    # Validate read_file: check file exists
    if name == "read_file":
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

    if name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"Error: Unknown tool '{name}'. Available tools: {available}"

    # Reject placeholder values like <filename>, <path>, etc.
    for key, val in args.items():
        if isinstance(val, str) and re.search(r'<[a-zA-Z_-]+>', val):
            return f"Error: Placeholder detected in '{key}': {val}. Use actual values, not placeholders."

    # Auto-convert relative paths to absolute for path arguments
    if 'path' in args and args['path']:
        p = args['path'].strip()
        if not p.startswith('/') and not p.startswith('~'):
            args['path'] = str(Path.cwd() / p)

    fn = TOOLS[name]["fn"]
    # Filter args to only include valid parameters for the function
    sig = inspect.signature(fn)
    valid_params = set(sig.parameters.keys())
    filtered_args = {k: v for k, v in args.items() if k in valid_params}
    return fn(**filtered_args)

def get_tool_schemas() -> list[dict]:
    """Get all tool schemas for Ollama."""
    return [t["schema"] for t in TOOLS.values()]
