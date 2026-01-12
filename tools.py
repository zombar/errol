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

def _clean_string(s: str) -> str:
    """Clean string by removing line number prefixes and trailing whitespace."""
    import re
    lines = s.split('\n')
    cleaned = []
    for line in lines:
        # Remove line number prefix (e.g., "123\t" or "  45\t")
        line = re.sub(r'^\s*\d+\t', '', line)
        # Strip trailing whitespace
        line = line.rstrip()
        cleaned.append(line)
    return '\n'.join(cleaned)


def _fix_indentation(old_string: str, new_string: str) -> str:
    """Ensure new_string lines have appropriate indentation matching old_string.

    Handles three common LLM mistakes:
    1. Forgot base offset but preserved relative structure -> add offset to all lines
    2. Got first line correct but forgot indent on subsequent lines -> fix those lines
    3. Excessive indentation (e.g., copying badly formatted code) -> normalize it
    """
    # Get base indentation from old_string's first line
    old_first_line = old_string.split('\n')[0]
    base_indent = len(old_first_line) - len(old_first_line.lstrip())

    # Detect indent character (tabs vs spaces) from the old string
    indent_char = '\t' if old_first_line.startswith('\t') else ' '
    base_whitespace = indent_char * base_indent if indent_char == '\t' else ' ' * base_indent

    # Get indentation of new_string's first line
    new_first_line = new_string.split('\n')[0]
    new_first_indent = len(new_first_line) - len(new_first_line.lstrip())

    # Calculate indent deficit (how much more indent we need on first line)
    indent_deficit = base_indent - new_first_indent

    # Max reasonable indentation - base + 12 spaces (3 nesting levels)
    max_indent = base_indent + 12

    # Handle single-line case
    if '\n' not in new_string:
        if new_string.strip():
            if indent_deficit > 0:
                # Under-indented - add spaces
                deficit_whitespace = indent_char * indent_deficit if indent_char == '\t' else ' ' * indent_deficit
                return deficit_whitespace + new_string
            elif new_first_indent > max_indent:
                # Over-indented - normalize to base
                return base_whitespace + new_string.lstrip()
        return new_string

    # Process new_string lines
    new_lines = new_string.split('\n')
    fixed_lines = []

    if indent_deficit > 0:
        # Case 1: First line is under-indented - add deficit to ALL lines
        deficit_whitespace = indent_char * indent_deficit if indent_char == '\t' else ' ' * indent_deficit
        for line in new_lines:
            if not line.strip():  # Empty or whitespace-only line
                fixed_lines.append(line)
            else:
                fixed_lines.append(deficit_whitespace + line)
    else:
        # Case 2 & 3: Fix under-indented or over-indented lines
        for i, line in enumerate(new_lines):
            if not line.strip():  # Empty or whitespace-only line
                fixed_lines.append(line)
            else:
                current_indent = len(line) - len(line.lstrip())
                if current_indent < base_indent:
                    # Under-indented - add base indent
                    fixed_lines.append(base_whitespace + line.lstrip())
                elif current_indent > max_indent:
                    # Over-indented - calculate relative indent from first line and normalize
                    if i == 0:
                        # First line over-indented - normalize to base
                        fixed_lines.append(base_whitespace + line.lstrip())
                    else:
                        # Subsequent line - preserve relative structure but cap it
                        relative_indent = current_indent - new_first_indent
                        normalized_indent = min(base_indent + relative_indent, max_indent)
                        norm_whitespace = indent_char * normalized_indent if indent_char == '\t' else ' ' * normalized_indent
                        fixed_lines.append(norm_whitespace + line.lstrip())
                else:
                    fixed_lines.append(line)

    return '\n'.join(fixed_lines)


# Unicode normalization map for common LLM substitutions
CHAR_NORMALIZE_MAP = {
    '\u201c': '"',  # Left double quotation mark
    '\u201d': '"',  # Right double quotation mark
    '\u2018': "'",  # Left single quotation mark
    '\u2019': "'",  # Right single quotation mark
    '\u00a0': ' ',  # Non-breaking space
    '\u2003': ' ',  # Em space
    '\u2002': ' ',  # En space
    '\u2009': ' ',  # Thin space
    '\u200b': '',   # Zero-width space (remove)
    '\u2013': '-',  # En dash
    '\u2014': '-',  # Em dash
    '\ufeff': '',   # BOM (remove)
}


def _find_indentation_agnostic_match(content: str, old_string: str) -> tuple:
    """Find multi-line match ignoring leading indentation differences.

    Returns (matched_string_from_file, count) or (None, 0) if not found.
    """
    old_lines = old_string.split('\n')
    if not old_lines:
        return (None, 0)

    # Strip leading whitespace from each line for comparison
    old_stripped = [line.lstrip() for line in old_lines]
    content_lines = content.split('\n')

    matches = []
    match_length = len(old_lines)

    for i in range(len(content_lines) - match_length + 1):
        candidate_lines = content_lines[i:i + match_length]
        candidate_stripped = [line.lstrip() for line in candidate_lines]

        if candidate_stripped == old_stripped:
            matched_str = '\n'.join(candidate_lines)
            matches.append(matched_str)

    if len(matches) == 1:
        return (matches[0], 1)
    elif len(matches) > 1:
        return (None, len(matches))
    return (None, 0)


def _normalize_whitespace(s: str) -> str:
    """Normalize whitespace for comparison.

    - Convert CRLF to LF
    - Convert tabs to 4 spaces
    - Strip trailing whitespace per line
    """
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    lines = s.split('\n')
    normalized = []
    for line in lines:
        line = line.rstrip()
        line = line.replace('\t', '    ')
        normalized.append(line)
    return '\n'.join(normalized)


def _find_whitespace_normalized_match(content: str, old_string: str) -> str:
    """Find old_string in content with whitespace normalization.

    Returns the actual string from content that matches, or None.
    """
    norm_old = _normalize_whitespace(old_string)
    norm_content = _normalize_whitespace(content)

    if norm_old not in norm_content:
        return None

    # Find position in normalized content
    norm_pos = norm_content.find(norm_old)
    norm_lines_before = norm_content[:norm_pos].count('\n')
    old_line_count = old_string.count('\n') + 1

    # Extract corresponding lines from original content
    content_lines = content.split('\n')
    matched_lines = content_lines[norm_lines_before:norm_lines_before + old_line_count]
    return '\n'.join(matched_lines)


def _normalize_unicode(s: str) -> str:
    """Normalize common unicode characters to ASCII equivalents."""
    for unicode_char, ascii_char in CHAR_NORMALIZE_MAP.items():
        s = s.replace(unicode_char, ascii_char)
    return s


def _find_unicode_normalized_match(content: str, old_string: str) -> str:
    """Find match with unicode normalization."""
    norm_old = _normalize_unicode(old_string)

    # If original already matches, use it
    if old_string in content:
        return old_string

    # Try normalized version
    if norm_old in content:
        pos = content.find(norm_old)
        return content[pos:pos + len(norm_old)]

    # Try normalizing both
    norm_content = _normalize_unicode(content)
    if norm_old in norm_content:
        pos = norm_content.find(norm_old)
        return content[pos:pos + len(norm_old)]

    return None


def _find_fuzzy_match(content: str, old_string: str, threshold: float = 0.90) -> tuple:
    """Find best fuzzy match using SequenceMatcher.

    Returns (matched_string, similarity_ratio, diff_operations) or None.
    """
    old_lines = old_string.split('\n')
    content_lines = content.split('\n')
    match_length = len(old_lines)

    best_match = None
    best_ratio = 0.0

    for i in range(len(content_lines) - match_length + 1):
        candidate = '\n'.join(content_lines[i:i + match_length])

        # Quick reject if lengths differ too much
        len_ratio = min(len(candidate), len(old_string)) / max(len(candidate), len(old_string), 1)
        if len_ratio < threshold:
            continue

        matcher = difflib.SequenceMatcher(None, old_string, candidate, autojunk=False)
        ratio = matcher.ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    if best_ratio >= threshold and best_match is not None:
        matcher = difflib.SequenceMatcher(None, old_string, best_match, autojunk=False)
        opcodes = matcher.get_opcodes()
        return (best_match, best_ratio, opcodes)

    return None


def _format_fuzzy_diff(old_string: str, matched: str, opcodes: list) -> str:
    """Format a human-readable diff showing exactly what differs."""
    lines = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        old_part = old_string[i1:i2]
        new_part = matched[j1:j2]
        if tag == 'replace':
            lines.append(f"  Expected {old_part!r}, found {new_part!r}")
        elif tag == 'delete':
            lines.append(f"  Extra in search: {old_part!r}")
        elif tag == 'insert':
            lines.append(f"  Missing from search: {new_part!r}")
    return '\n'.join(lines[:5])  # Limit to 5 differences


def _generate_match_diagnostic(content: str, old_string: str) -> str:
    """Generate detailed diagnostic for failed match."""
    diagnostics = []

    if '\t' in old_string and '\t' not in content:
        diagnostics.append("Search contains tabs but file uses spaces")
    elif '\t' in content and '\t' not in old_string:
        diagnostics.append("File contains tabs but search uses spaces")

    if '\r' in old_string:
        diagnostics.append("Search contains CRLF line endings")

    unicode_chars = [c for c in old_string if ord(c) > 127]
    if unicode_chars:
        diagnostics.append(f"Search contains unicode: {set(unicode_chars)}")

    # Find similar lines
    first_line = old_string.split('\n')[0]
    matches = difflib.get_close_matches(
        first_line.strip(),
        [l.strip() for l in content.splitlines()],
        n=1,
        cutoff=0.6
    )
    if matches:
        diagnostics.append(f"Similar line in file: {matches[0]!r}")

    return '\n'.join(diagnostics) if diagnostics else "No similar content found"


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

        # If not found, try with cleaned strings (no line numbers, no trailing whitespace)
        if count == 0:
            cleaned_old = _clean_string(old_string)
            cleaned_new = _clean_string(new_string)
            # Normalize file content too (strip trailing whitespace per line)
            content_normalized = '\n'.join(line.rstrip() for line in content.split('\n'))
            count = content_normalized.count(cleaned_old)
            if count > 0:
                # Use cleaned versions with indentation fix
                old_string = cleaned_old
                new_string = _fix_indentation(cleaned_old, cleaned_new)
                content = content_normalized

        if count == 0:
            # Try to find line with matching content but different indentation
            search_stripped = old_string.strip()
            matching_lines = []
            for file_line in content.splitlines():
                if file_line.strip() == search_stripped:
                    matching_lines.append(file_line)
            if len(matching_lines) == 1:
                # Found exactly one match - use the actual line from the file
                old_string = matching_lines[0]
                # Apply indentation fix based on the actual line's indentation
                new_string = _fix_indentation(old_string, new_string)
                count = 1
            elif len(matching_lines) > 1 and not replace_all:
                # Use the first match when there are multiple
                old_string = matching_lines[0]
                new_string = _fix_indentation(old_string, new_string)
                count = 1

        # Try indentation-agnostic multi-line match
        if count == 0 and '\n' in old_string:
            matched, match_count = _find_indentation_agnostic_match(content, old_string)
            if match_count == 1:
                old_string = matched
                new_string = _fix_indentation(old_string, new_string)
                count = 1
            elif match_count > 1 and not replace_all:
                return f"Error: {match_count} matches found in {path}. Provide more context for uniqueness."

        # Try whitespace-normalized match (tabs vs spaces, CRLF vs LF)
        if count == 0:
            matched = _find_whitespace_normalized_match(content, old_string)
            if matched:
                old_string = matched
                new_string = _fix_indentation(old_string, new_string)
                count = 1

        # Try unicode-normalized match (smart quotes, special spaces)
        if count == 0:
            matched = _find_unicode_normalized_match(content, old_string)
            if matched:
                old_string = matched
                count = 1

        # Last resort: fuzzy match - show what differs instead of silently failing
        if count == 0:
            fuzzy_result = _find_fuzzy_match(content, old_string, threshold=0.90)
            if fuzzy_result:
                matched, ratio, opcodes = fuzzy_result
                diff_info = _format_fuzzy_diff(old_string, matched, opcodes)
                return f"Error: Exact string not found. Similar match ({ratio:.0%}):\n{diff_info}\nUse the exact text from the file."

        if count == 0:
            # Generate detailed diagnostic for failed match
            diagnostic = _generate_match_diagnostic(content, old_string)
            first_line = old_string.split('\n')[0][:80]
            return f"Error: String not found in {path}.\nLooking for: {first_line!r}\n{diagnostic}"
        # Fix indentation for multi-line edits where LLM forgot proper spacing
        new_string = _fix_indentation(old_string, new_string)
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced_msg = f" ({count} occurrences)" if count > 1 else ""
        else:
            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)
            replaced_msg = f" (first of {count})" if count > 1 else ""
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
                "description": "Replace a string in a file with new content. Replaces first occurrence by default, or all if replace_all=true.",
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
    }
}

def validate_tool_call(name: str, args: dict) -> Optional[str]:
    """Pre-validate a tool call. Returns error string if invalid, None if OK."""
    import re

    # Check tool exists
    if name not in TOOLS:
        tool_list = "\n".join([
            "  - read_file(path): Read file contents",
            "  - write_file(path, content): Create/overwrite file",
            "  - edit_file(path, old_string, new_string): Replace text in file",
            "  - bash(command): Run shell command",
            "  - glob(pattern): Find files by name pattern",
            "  - grep(pattern, path): Search text in files",
        ])
        return f"Unknown tool '{name}'. Use ONLY these tools:\n{tool_list}"

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
                # Try fallback matching with cleaned strings (same as edit_file does)
                if count == 0:
                    cleaned_old = _clean_string(old_string)
                    content_normalized = '\n'.join(line.rstrip() for line in content.split('\n'))
                    count = content_normalized.count(cleaned_old)
                if count == 0:
                    # Try to find the string ignoring leading whitespace differences
                    search_stripped = old_string.strip()
                    for line in content.splitlines():
                        if line.strip() == search_stripped:
                            count = 1  # Found with different indentation, let edit_file handle it
                            break
                # Try indentation-agnostic multi-line match
                if count == 0 and '\n' in old_string:
                    _, match_count = _find_indentation_agnostic_match(content, old_string)
                    if match_count >= 1:
                        count = match_count
                # Try whitespace-normalized match
                if count == 0:
                    matched = _find_whitespace_normalized_match(content, old_string)
                    if matched:
                        count = 1
                # Try unicode-normalized match
                if count == 0:
                    matched = _find_unicode_normalized_match(content, old_string)
                    if matched:
                        count = 1
                # Check for fuzzy match (let edit_file show the diff)
                if count == 0:
                    fuzzy_result = _find_fuzzy_match(content, old_string, threshold=0.90)
                    if fuzzy_result:
                        count = 1  # Let edit_file handle showing the diff
                if count == 0:
                    first_line = old_string.split('\n')[0][:80]
                    return f"String not found in {path}. Looking for: {first_line!r}"
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
        tool_list = "\n".join([
            "  - read_file(path): Read file contents",
            "  - write_file(path, content): Create/overwrite file",
            "  - edit_file(path, old_string, new_string): Replace text in file",
            "  - bash(command): Run shell command",
            "  - glob(pattern): Find files by name pattern",
            "  - grep(pattern, path): Search text in files",
        ])
        return f"Error: Unknown tool '{name}'. Use ONLY these tools:\n{tool_list}"

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
