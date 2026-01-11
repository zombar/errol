"""Task router - classifies tasks and selects appropriate model."""
import re
import json
from typing import Tuple
from llm import OllamaClient

# Pattern-based fast routing (no LLM call needed)
PATTERNS = {
    "small": [
        r"\btodo\b", r"\btask\b", r"\blist\b", r"\bread\b", r"\bshow\b",
        r"\bcat\b", r"\bls\b", r"\bfind\b", r"\bglob\b", r"\bsearch\b"
    ],
    "medium": [
        r"\bexplain\b", r"\bwhat is\b", r"\bwhat does\b", r"\bhow does\b",
        r"\bdescribe\b", r"\bsummarize\b", r"\breview\b"
    ],
    "large": [
        r"\bcreate\b", r"\bimplement\b", r"\bbuild\b", r"\bwrite\b",
        r"\brefactor\b", r"\bgenerate\b", r"\badd feature\b", r"\bnew\b",
        r"\bfix\b", r"\bdebug\b", r"\boptimize\b"
    ]
}

CLASSIFICATION_PROMPT = """Classify this task into one complexity level.

Task: {task}

Respond with ONLY a JSON object:
{{"complexity": "simple|moderate|complex", "category": "code_gen|code_edit|explanation|debug|file_ops|todo|question"}}

Guidelines:
- simple: file reads, listing, todos, simple questions
- moderate: explanations, small edits, known fixes
- complex: new code, features, refactoring, debugging"""


class Router:
    def __init__(self, client: OllamaClient, models: dict):
        self.client = client
        self.models = models  # {"small": "...", "medium": "...", "large": "..."}

    def classify(self, task: str) -> Tuple[str, str]:
        """Classify task and return (model_name, category)."""
        # Try pattern matching first
        task_lower = task.lower()

        for tier, patterns in PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    return self.models[tier], tier

        # Fall back to LLM classification using small model
        return self._llm_classify(task)

    def _llm_classify(self, task: str) -> Tuple[str, str]:
        """Use small model to classify task."""
        messages = [
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(task=task)}
        ]

        try:
            resp = self.client.chat_sync(self.models["small"], messages)
            content = resp.get("message", {}).get("content", "")

            # Parse JSON from response
            match = re.search(r'\{[^}]+\}', content)
            if match:
                data = json.loads(match.group())
                complexity = data.get("complexity", "moderate")
                category = data.get("category", "question")

                tier = {"simple": "small", "moderate": "medium", "complex": "large"}.get(
                    complexity, "medium"
                )
                return self.models[tier], category
        except Exception as e:
            print(f"[router] Classification failed: {e}, defaulting to medium")

        return self.models["medium"], "question"
