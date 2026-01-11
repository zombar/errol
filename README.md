# Errol

A minimal, self-modifying MoE (Mixture of Experts) local LLM agent using Ollama.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:14b

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
# Interactive chat
errol chat

# Single task
errol chat "read errol.py and explain it"
errol chat "add a --verbose flag to the chat command"

# Todos
errol todo add "implement caching"
errol todo list
errol todo start <id>
errol todo done <id>
errol todo rm <id>
errol todo clear

# Check available models
errol models

# Validate own source code
errol self-check
```

## Model Routing

| Tier   | Model              | Use Case                          |
|--------|--------------------|------------------------------------|
| Small  | qwen2.5:3b         | Routing, todos, file reads         |
| Medium | qwen2.5:7b         | Explanations, small edits          |
| Large  | qwen2.5-coder:14b  | Code generation, refactors         |

## Self-Modification

Errol knows its own source location and can modify itself:

```bash
errol chat "add a new tool called 'search_code' to tools.py"
```

All file changes show a git-style diff and require confirmation before applying.

## Configuration

Edit `config.yaml` to change models or settings:

```yaml
ollama:
  host: http://localhost:11434
  timeout: 300

models:
  small: qwen2.5:3b
  medium: qwen2.5:7b
  large: qwen2.5-coder:14b

agent:
  max_turns: 75
```
