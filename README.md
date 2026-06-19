<div align="center">

# Terminal AI Agent

**A fast, colorful AI agent for your terminal — powered by free models from 5 providers, with a built-in coding agent mode.**

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-dea584?logo=rust)](https://www.rust-lang.org)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-yesk993--ops/Terminal--AI--Agent-181717?logo=github)](https://github.com/yesk993-ops/Terminal-AI-Agent)

```text
────────────────────────────────────────────────────────────────────────────────
Docker is a containerization platform that allows developers to package, ship,
and run applications in containers. Containers are lightweight and portable,
allowing developers to deploy applications quickly and efficiently.

Key Benefits:
  Lightweight — containers share the host OS kernel
  Portable   — run anywhere that supports containers
  Isolated   — secure by design
────────────────────────────────────────────────────────────────────────────────
```

</div>

## Features

- ⚡ **Parallel model racing** — queries across all providers simultaneously; fastest response wins
- 🤖 **Coding agent mode** (`--code`) — read, write, edit files; run bash commands; search code — all via natural language
- 🎨 **Color-coded output** — bold green for key terms, green for headings, code blocks, inline commands, gold for acronyms
- 🆓 **Free models, no keys required** — built-in OpenCode gateway with zero-config free models
- 🔑 **5 providers** — OpenRouter, Groq, Google Gemini, NVIDIA NIM, and local OpenCode gateway
- 💬 **Conversation memory** — remembers last 6 turns, persisted across restarts
- 📋 **Copy-friendly** — top/bottom border only, no side walls for easy copy-paste
- 🔄 **REPL & single-shot** — interactive mode with `ask` or one-liner from shell
- 🧵 **Graceful shutdown** — Ctrl+C saves conversation before exit

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/yesk993-ops/Terminal-AI-Agent/main/setup.sh | bash
```

The script auto-detects your OS, installs Rust, Node.js, and the OpenCode gateway (free models with no API key), builds the project, and installs the binary. After it finishes:

```bash
ask "What is Rust?"
```

Works immediately — no API keys needed thanks to the built-in OpenCode gateway.

> For detailed platform-specific instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Single query
```bash
terminal_ai_agent "Explain monads in functional programming"
```

### Coding agent mode
```bash
terminal_ai_agent --code "create a sample Dockerfile for a Rust web app"
```

The coding agent can read, write, edit files, run bash commands, search code with grep, and glob for files — all through natural language.

### Custom temperature
```bash
terminal_ai_agent --temp 0.7 "Write a haiku about Rust"
```

### REPL mode
```bash
terminal_ai_agent
> ask What is AWS Auto Scaling?
> ask create a sample Dockerfile
> exit
```

### Coding agent REPL
```bash
terminal_ai_agent --code
> ask find all Rust files with unsafe blocks
> ask refactor main.rs to use anyhow
> exit
```

### Shell shortcut
The setup script installs an `ask` wrapper at `/usr/local/bin/ask`. Alternatively, add to `~/.bashrc`:

```bash
ask() { terminal_ai_agent "$@"; }
```

Then:
```bash
ask What is the capital of France?
```

## How it works

```
                     ┌──────────────────────┐
                     │  OpenRouter          │
                     │  Groq                │
  ┌──────────┐       │  Google Gemini       │     ┌─────────┐
  │  Query   │ ────▶ │  NVIDIA NIM          │ ──▶ │  Fastest │
  │          │       │  OpenCode Gateway    │     │  answer  │
  │  user    │       │  (free, no key)      │     │   wins   │
  │  types   │       │                      │     │          │
  └──────────┘       └──────────────────────┘     └─────────┘
                     All models race in parallel
                     (10s timeout each)
```

The agent fires requests to every configured provider simultaneously. The first model to return a valid response wins. If all fail, a helpful summary shows which errors occurred and suggests missing API keys.

### Coding agent flow

```
  ┌──────────┐     ┌──────────────────────┐     ┌───────────┐
  │  User    │ ──▶ │  Model races across  │ ──▶ │ Tool call │
  │  prompt  │     │  all 5 providers     │     │ detected? │
  └──────────┘     └──────────────────────┘     ─────┬──────
                                                      │
                                           yes        │        no
                                            ┌─────────┘
                                            ▼
                                     ┌────────────┐
                                     │  Execute    │
                                     │  tool       │◀──── feedback loop
                                     │  (bash,     │      (max 25 iters)
                                     │   read,     │
                                     │   write,    │
                                     │   grep,     │
                                     │   glob,     │
                                     │   edit)     │
                                     └────────────┘
```

The coding agent (`--code`) uses text-based tool calling with `<tool_call>` tags — works with any model, no API-level function calling required.

## Providers

| Provider | Env var | Key required | Models |
|---|---|---|---|
| OpenCode Gateway | — | **No** | `big-pickle`, `gpt-5-nano`, `minimax-m2.5-free`, `nemotron-3-super-free` |
| OpenRouter | `OPENROUTER_API_KEY` | Yes | `deepseek/chat`, `gemini-2.0-flash`, `llama-4-maverick`, `qwen3`, `mistral-small`, `phi-4`, `gemma-4`, `owl-alpha` |
| Groq | `GROQ_API_KEY` | Yes | `llama-3.3-70b`, `mixtral-8x7b`, `llama-3.1-8b` |
| Google Gemini | `GOOGLE_API_KEY` | Yes | `gemini-1.5-flash`, `gemini-2.0-flash-lite` |
| NVIDIA NIM | `NVIDIA_API_KEY` | Yes | `nemotron-3-super`, `meta/llama-3.1-8b-instruct` |

The OpenCode gateway runs locally on `http://127.0.0.1:8083` and needs no API key. The setup script installs it automatically and creates a systemd/launchd service for autostart.

### Getting API keys

| Provider | Where to get |
|---|---|
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) |
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) |

Keys are **never** stored in source code — only read from environment variables at runtime.

## Color reference

| Element | ANSI code | Sample |
|---|---|---|
| **Bold key terms** `**text**` | `\x1b[1;32m` (bold green) | **key concept** |
| Headings `### Title` | `\x1b[32m` (green) | Section heading |
| Code blocks `` ``` `` | `\x1b[0;32m` (green) | Multi-line code |
| Inline commands `` `cmd` `` | `\x1b[0;32m` (green) | Commands, flags, paths |
| Shell prompts `$ cmd` | `\x1b[0;32m` (green) | Command-line examples |
| Acronyms `API (...)` | `\x1b[38;5;220m` (gold) | Acronym definitions |
| Border | `\x1b[36m` (cyan) | Top/bottom horizontal rule |

## Project structure

```
src/
  lib.rs    — core library: types, API calls, formatting, conversation, tests
  main.rs   — binary entry point: CLI parsing, REPL loop
  tools.rs  — coding agent tools: read, write, edit, bash, grep, glob, list_dir
setup.sh    — one-command installer (auto-detects OS, installs deps + builds)
INSTALL.md  — platform-specific installation guide
```

## Build from source

```bash
git clone https://github.com/yesk993-ops/Terminal-AI-Agent.git
cd Terminal-AI-Agent
cargo build --release
./target/release/terminal_ai_agent "Hello"
```

## Tests

```bash
cargo test
```

## License

[MIT](LICENSE)
