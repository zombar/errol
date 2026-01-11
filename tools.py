"""Tool registry and implementations."""
import os
import glob as globlib
import subprocess
import difflib
from pathlib import Path

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

def read_file(path: str, offset: int = None, limit: int = None) -> str:
    """Read a file with optional line offset and limit."""
    try:
        # Handle None values from JSON null
        if offset is None:
            offset = 0
        if limit is None:
            limit = 2000

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

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in file."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text()
        count = content.count(old_string)

        if count == 0:
            return f"Error: String not found in file"
        if count > 1:
            return f"Error: String found {count} times, must be unique"

        new_content = content.replace(old_string, new_string)
        p.write_text(new_content)
        return f"Edited {path}: applied changes"
    except Exception as e:
        return f"Error editing file: {e}"

def run_bash(command: str, timeout: int = 120) -> str:
    """Execute a bash command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        output = result.stdout + result.stderr
        if len(output) > 10000:
            output = output[:10000] + "\n... (truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"

def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern."""
    try:
        base = Path(path).expanduser().resolve()
        matches = list(base.glob(pattern))
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
                "description": "Replace a unique string in a file with new content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "old_string": {"type": "string", "description": "String to find (must be unique)"},
                        "new_string": {"type": "string", "description": "Replacement string"}
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

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name with arguments."""
    if name not in TOOLS:
        return f"Error: Unknown tool '{name}'"
    return TOOLS[name]["fn"](**args)

def get_tool_schemas() -> list[dict]:
    """Get all tool schemas for Ollama."""
    return [t["schema"] for t in TOOLS.values()]
