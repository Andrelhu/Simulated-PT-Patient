# Ana Agent v0.2 (claude_ver)

Simulated PT patient console agent. No OpenAI dependency.

## Backends

### Ollama (default — 100% local, private)

1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull llama3.2:8b
   ```
2. Run:
   ```bash
   python -m ana_agent --model llama3.2:8b
   ```

### Anthropic Claude API

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Run:
```bash
python -m ana_agent --model claude-3-5-haiku-20241022 --backend claude
```

## Options

```
--model           Model name (required)
--backend         ollama | claude  (default: ollama)
--session-id      Session label for logs (default: console)
--no-guardrail    Skip the guardrail stage (single-pass, faster)
```

## Spec files (required)

```
data/specs/behavior.txt    Ana's behavioral rules
data/specs/character.txt   Ana's character description
```

## Session logs

Saved to `data/session_logs/` after each session.
