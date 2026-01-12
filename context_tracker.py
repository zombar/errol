"""Context tracker for maintaining task state across agent turns."""
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class FileInfo:
    """Information about a read file."""
    path: str
    excerpt: str  # First N lines for quick reference
    line_range: tuple[int, int]  # (offset, offset+limit)
    summary: Optional[str] = None  # LLM-generated summary


@dataclass
class SearchResult:
    """Information about a search operation."""
    tool: str  # "grep" or "glob"
    pattern: str
    results: str  # Truncated results


class ContextTracker:
    """Tracks gathered context and generates reminders for the LLM."""

    # Rough chars-per-token estimate
    CHARS_PER_TOKEN = 4

    def __init__(self, task: str, context_limit: int = 8192):
        self.task = task
        self.context_limit = context_limit  # Token limit for the model

        # Tracked state
        self.files_read: dict[str, FileInfo] = {}
        self.searches: list[SearchResult] = []
        self.edits_made: list[tuple[str, str]] = []  # (path, description)
        self.commands_run: list[tuple[str, str]] = []  # (cmd, output_summary)

        # Summarization state
        self.turn_count = 0
        self.last_summary_turn = 0
        self.llm_summary: Optional[str] = None
        self.summary_interval = 10  # Summarize every N turns
        self.context_threshold = 0.70  # Summarize when context > 70% full

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate from message content."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            # Tool calls also consume tokens
            if "tool_calls" in msg:
                total_chars += len(str(msg["tool_calls"]))
        return total_chars // self.CHARS_PER_TOKEN

    def should_summarize(self, messages: list[dict]) -> bool:
        """Check if we need to generate an LLM summary."""
        # Every N turns since last summary
        turns_since_summary = self.turn_count - self.last_summary_turn
        if turns_since_summary >= self.summary_interval:
            return True

        # Or if context is getting full
        token_estimate = self.estimate_tokens(messages)
        if token_estimate > self.context_limit * self.context_threshold:
            return True

        return False

    def record_tool_result(self, name: str, args: dict, result: str):
        """Record a tool execution for tracking."""
        self.turn_count += 1

        if name == "read_file":
            path = args.get("path", "")
            offset = args.get("offset", 0) or 0
            limit = args.get("limit", 2000) or 2000

            # Store first 30 lines as excerpt
            lines = result.split('\n')[:30]
            excerpt = '\n'.join(lines)
            if len(result.split('\n')) > 30:
                excerpt += "\n... (truncated)"

            self.files_read[path] = FileInfo(
                path=path,
                excerpt=excerpt,
                line_range=(offset, offset + limit),
                summary=None
            )

        elif name in ("grep", "glob"):
            # Store search results (already concise)
            self.searches.append(SearchResult(
                tool=name,
                pattern=args.get("pattern", ""),
                results=result[:500] + ("..." if len(result) > 500 else "")
            ))

        elif name == "edit_file":
            path = args.get("path", "")
            old = args.get("old_string", "")[:50]
            new = args.get("new_string", "")[:50]
            self.edits_made.append((path, f"replaced '{old}...' with '{new}...'"))

        elif name == "write_file":
            path = args.get("path", "")
            size = len(args.get("content", ""))
            self.edits_made.append((path, f"wrote {size} bytes"))

        elif name == "bash":
            cmd = args.get("command", "")[:80]
            output = result[:200] + ("..." if len(result) > 200 else "")
            self.commands_run.append((cmd, output))

    def is_duplicate_read(self, path: str, offset: int = 0) -> bool:
        """Check if we've already read this file at this offset."""
        if path not in self.files_read:
            return False
        info = self.files_read[path]
        # Consider duplicate if same file and overlapping range
        return info.line_range[0] == offset

    def is_duplicate_search(self, tool: str, pattern: str) -> bool:
        """Check if we've already done this exact search."""
        for s in self.searches:
            if s.tool == tool and s.pattern == pattern:
                return True
        return False

    def get_simple_reminder(self) -> str:
        """Generate a quick reminder prompting the model to reflect and continue."""
        lines = [
            "─" * 40,
            "## CONTEXT CHECKPOINT",
            f"**Original task:** {self.task}",
            ""
        ]

        if self.files_read:
            lines.append("**Files you have read (do NOT re-read):**")
            for path, info in self.files_read.items():
                range_str = f"lines {info.line_range[0]}-{info.line_range[1]}"
                if info.summary:
                    lines.append(f"  • {path} ({range_str}): {info.summary}")
                else:
                    lines.append(f"  • {path} ({range_str})")

        if self.searches:
            lines.append("")
            lines.append("**Searches performed:**")
            for s in self.searches[-5:]:  # Last 5 searches
                lines.append(f"  • {s.tool}('{s.pattern}')")

        if self.edits_made:
            lines.append("")
            lines.append("**Edits completed:**")
            for path, desc in self.edits_made[-5:]:
                lines.append(f"  • {path}: {desc}")

        lines.extend([
            "",
            "## REQUIRED: Before your next action, briefly state:",
            "1. What key information have you found relevant to the task?",
            "2. What specific change will you make next and why?",
            "",
            "Then proceed with the tool call. Do NOT re-read any files listed above.",
            "─" * 40
        ])

        return "\n".join(lines)

    def get_summarization_prompt(self) -> str:
        """Generate prompt for LLM to summarize gathered context."""
        sections = [f"Task: {self.task}", ""]

        if self.files_read:
            sections.append("Files read:")
            for path, info in self.files_read.items():
                sections.append(f"\n### {path}")
                sections.append(info.excerpt)

        if self.searches:
            sections.append("\nSearch results:")
            for s in self.searches:
                sections.append(f"{s.tool}('{s.pattern}'):\n{s.results}")

        if self.edits_made:
            sections.append("\nEdits made:")
            for path, desc in self.edits_made:
                sections.append(f"  - {path}: {desc}")

        prompt = "\n".join(sections)
        prompt += """

---
Summarize the above context in 100-150 words:
1. What key information was found relevant to the task?
2. What specific code/lines need to be changed?
3. What is the next concrete step?

Be specific - mention line numbers, function names, variable names."""

        return prompt

    def store_summary(self, summary: str):
        """Store LLM-generated summary."""
        self.llm_summary = summary
        self.last_summary_turn = self.turn_count

    def get_full_reminder(self) -> str:
        """Get reminder with LLM summary if available."""
        if self.llm_summary:
            lines = [
                "─" * 40,
                "## CONTEXT SUMMARY (from previous analysis)",
                self.llm_summary,
                "",
                f"## Original task: {self.task}",
                "",
                "## Files read (do not re-read):",
            ]
            for path in self.files_read.keys():
                lines.append(f"  • {path}")
            lines.append("─" * 40)
            return "\n".join(lines)
        else:
            return self.get_simple_reminder()
