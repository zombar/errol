# Errol

A minimal, self-modifying local LLM agent using Ollama.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
# The project now uses the gpt-oss:20b model for all tiers
ollama pull gpt-oss:20b

# Start Ollama
ollama serve
```

## Installation

To run `errol` from anywhere:

```bash
# Make the wrapper script executable
chmod +x /path/to/errol/errol

# Add to your shell profile (~/.zshrc or ~/.bashrc)
export PATH="$PATH:/path/to/errol"

# Reload your shell
source ~/.zshrc  # or ~/.bashrc
```

Replace `/path/to/errol` with the actual path to the errol directory.

## Usage

```bash
# Interactive mode (default)
errol

# Single task
errol chat "read errol.py and explain it"
errol chat "add a --verbose flag to the chat command"

# Check available models
errol models

# Validate own source code
errol self-check
```

## Model Routing

| Tier   | Model              | Use Case                           |
|--------|--------------------|------------------------------------|
| Small  | gpt-oss:20b        | Routing, file reads                |
| Medium | gpt-oss:20b        | Explanations, small edits          |
| Large  | gpt-oss:20b        | Code generation, refactors         |

## Self-Modification

Errol knows its own source location and can modify itself:

```bash
errol chat "add a new tool called 'search_code' to tools.py"
```

All file changes show a git‑style diff and require confirmation before applying.

## Configuration

Edit `config.yaml` to change models or settings:

```yaml
ollama:
  host: http://localhost:11434
  timeout: 300

models:
  small: gpt-oss:20b
  medium: gpt-oss:20b
  large: gpt-oss:20b

agent:
  max_turns: 75
  show_model_picker: false
```
