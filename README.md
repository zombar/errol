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

## Usage

```bash
# Interactive chat
python3 errol.py chat

# Single task
python3 errol.py chat "read errol.py and explain it"
python3 errol.py chat "add a --verbose flag to the chat command"

# Todos
python3 errol.py todo add "implement caching"
python3 errol.py todo list
python3 errol.py todo start <id>
python3 errol.py todo done <id>
python3 errol.py todo rm <id>
python3 errol.py todo clear

# Check available models
python3 errol.py models

# Validate own source code
python3 errol.py self-check
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
python3 errol.py chat "add a new tool called 'search_code' to tools.py"
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
  max_turns: 20
```
