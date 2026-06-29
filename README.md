<div align="center">

# Terminal AI Agent

**A fast, colorful AI agent for your terminal — powered by free models from 5 providers, with ChatGPT-level features.**

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
- 🎯 **Streaming responses** — tokens appear in real-time as the model generates (ChatGPT-like experience)
- 🤖 **Coding agent mode** (`--code`) — read, write, edit files; run bash commands; search code — all via natural language
- 🧠 **Chain-of-Thought reasoning** — models reason step-by-step for deeper, more accurate answers
- 🔄 **Self-reflection** — after answering, the agent critiques and improves its own response in background
- 💡 **Smart follow-up suggestions** — type `suggest` in REPL to see 2-3 relevant follow-up questions
- 📦 **Response caching** — identical queries return instantly on repeat
- 📁 **Project-aware context** — coding agent automatically scans your project structure
- 🎨 **Color-coded output** — bold for key terms, headings, code blocks, inline commands, acronyms
- 🆓 **Free models, no keys required** — built-in OpenCode gateway with zero-config free models
- 🔑 **5 providers** — NVIDIA NIM, Groq, OpenRouter, Google Gemini, and local OpenCode gateway
- 💬 **Conversation memory** — remembers context across turns, persisted across restarts
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

### Streaming response
When `GROQ_API_KEY` is set, tokens stream in real-time automatically:
```bash
terminal_ai_agent "What is the difference between Docker and Podman?"
```

### Coding agent mode (with alias)
```bash
code "create a sample Dockerfile for a Rust web app"
# or without alias:
terminal_ai_agent --code "create a sample Dockerfile for a Rust web app"
```

### Custom temperature
```bash
terminal_ai_agent --temp 0.7 "Write a haiku about Rust"
```

### REPL mode
```bash
terminal_ai_agent
> ask What is AWS Auto Scaling?
> ask create a sample Dockerfile
> suggest                    # see follow-up questions
> exit
```

### Coding agent REPL (with alias)
```bash
code
> ask find all Rust files with unsafe blocks
> ask refactor main.rs to use anyhow
> exit
```

### Shell aliases

The setup script installs an `ask` wrapper at `/usr/local/bin/ask`. For quick access to both modes, add these to `~/.bashrc` or `~/.zshrc`:

```bash
# Query mode (general Q&A)
ask()   { terminal_ai_agent "$@"; }

# Code agent mode (file operations, bash, search)
code()  { terminal_ai_agent --code "$@"; }
```

Then use them like this:
```bash
# Query mode — general questions
ask "What is the capital of France?"

# Code agent — file operations, bash, project tasks
code "create a Dockerfile for a Rust web app"
code "find all Rust files with unsafe blocks"
```

## Environment variables

### Setting API keys

The agent reads API keys from environment variables at runtime — never stored in source code.

#### Per-user (recommended)
Add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`:
```bash
export NVIDIA_API_KEY="nvapi-..."
export GROQ_API_KEY="gsk_..."
export OPENROUTER_API_KEY="sk-or-..."
```

Reload:
```bash
source ~/.bashrc
```

#### Global (all users)
Add to `/etc/environment` or `/etc/profile.d/terminal-ai-agent.sh`:
```bash
# /etc/profile.d/terminal-ai-agent.sh
export NVIDIA_API_KEY="nvapi-..."
export GROQ_API_KEY="gsk_..."
```

### Available variables

| Variable | Required | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | Recommended | Primary production provider (1000+ RPM, no rate limits) |
| `GROQ_API_KEY` | Optional | Fastest provider (~1-3s), enables **streaming** and **self-reflection** |
| `OPENROUTER_API_KEY` | Optional | Fallback via free models from 5 providers |
| `NVIDIA_QWEN_API_KEY` | Optional | Dedicated Qwen model via NVIDIA NIM |
| `OPENROUTER_MODEL` | Optional | Override the default free model list with a specific model |

### Setting a specific model
```bash
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
```

### Verify keys are set
```bash
echo "NVIDIA: ${NVIDIA_API_KEY:+set (${#NVIDIA_API_KEY} chars)}"
echo "Groq: ${GROQ_API_KEY:+set (${#GROQ_API_KEY} chars)}"
```

## Providers

| Provider | Env var | Key required | Models | Rate limits |
|---|---|---|---|---|
| **NVIDIA NIM** ⭐ | `NVIDIA_API_KEY` | Yes | `deepseek-ai/deepseek-v4-pro`, `mistralai/mistral-small-4-119b-2603`, `meta/llama-3.1-8b-instruct` | **1000+ RPM** (no 429s) |
| Groq | `GROQ_API_KEY` | Yes | `llama-3.3-70b-versatile` | 30 RPM (free tier) |
| OpenRouter | `OPENROUTER_API_KEY` | Yes | `:free` models from 5 providers | 1-5 RPM (free tier) |
| OpenCode Gateway | — | **No** | `big-pickle`, `gpt-5-nano` | varies |

> **⭐ NVIDIA NIM is the recommended primary provider.** Production models with 1000+ RPM rate limits mean you will **never** see HTTP 429 errors. Get a free key at [build.nvidia.com](https://build.nvidia.com).

**Groq is recommended as a secondary provider** — it's the fastest (~1-3s) and enables streaming responses and self-reflection.

### Getting API keys

| Provider | Where to get | Recommended? |
|---|---|---|
| **NVIDIA NIM** | [build.nvidia.com](https://build.nvidia.com) | **✅ Primary (no rate limits)** |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | Optional fallback |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | Optional fallback |
| OpenCode Gateway | Built-in, no key needed | Zero-config free tier |

## How it works

### Query mode

```
                      ┌──────────────────────┐
                      │  Response Cache       │── hit → instant return
                      └──────────┬───────────┘
                                 │ miss
                                 ▼
  ┌──────────┐       ┌──────────────────────┐     ┌──────────────────┐
  │  Query   │ ────▶ │  Streaming (Groq)    │──▶ │  Real-time tokens │
  │          │       │  if key set           │     │  + horizontal    │
  │  user    │       └──────────────────────┘     │  rule when done   │
  │  types   │                                         │
  │          │       ┌──────────────────────┐          │
  └──────────┘───▶   │  Parallel racing     │──▶  │  First response  │
                     │  (fallback)          │     │  wins → display  │
                     │  NVIDIA + Groq + OR  │     └────────┬─────────┘
                     └──────────────────────┘              │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │  Background tasks:   │
                                                  │  • Self-reflection   │
                                                  │    (improve response)│
                                                  │  • Follow-up         │
                                                  │    suggestions       │
                                                  └──────────────────────┘
```

The agent first checks the **response cache**. If the same query was asked before, it returns instantly.

If `GROQ_API_KEY` is set, it tries **streaming** from Groq — tokens appear as the model generates them, similar to ChatGPT. After streaming finishes, background tasks improve the response further.

If streaming is unavailable or fails, the agent falls back to **parallel model racing** — all providers queried simultaneously via `FuturesUnordered`. The first valid response wins.

After every response, two background tasks run:
1. **Self-reflection** — a fast model (Groq) critiques the response and produces an improved version, silently updating the conversation history
2. **Follow-up suggestions** — generates 2-3 relevant follow-up questions, accessible via the `suggest` REPL command

### Coding agent flow

```
  ┌──────────┐     ┌──────────────────────┐     ┌──────────────────────┐
  │  User    │ ──▶ │  Project context     │──▶ │  Model races across  │
  │  prompt  │     │  injected (file tree) │     │  all providers       │
  └──────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │  Tool call detected? │
                                                  │  (native or text)    │
                                                  └──────┬───────┬───────┘
                                                    yes   │       │  no
                                                  ┌───────┘       └──────▶  Print
                                                  ▼                       response
                                      ┌──────────────────────────┐
                                      │  Duplicate check         │
                                      │  (skip if same tool+args) │
                                      │  + shell quoting fix     │
                                      └──────────┬───────────────┘
                                                  ▼
                                      ┌──────────────────────┐
                                      │  Execute tool         │
                                      │  (bash, read, write,  │
                                      │   grep, glob, edit)   │
                                      └──────────┬───────────┘
                                                  │
                                                  ▼
                                      ┌──────────────────────┐
                                      │  Inject result +      │
                                      │  remaining checklist  │
                                      │  Trim context if      │
                                      │  > 50 messages        │
                                      └──────────┬───────────┘
                                                  │
                                                  ▼
                                        Loop (max 40 iters)
                                        until no tool call
```

The coding agent (`--code`) follows a structured workflow:

1. **Project context injection** — before processing, the agent scans your current directory and injects a project summary (files, line counts) as context
2. **Model query** — all providers race to produce a tool call or answer
3. **Duplicate detection** — repeated identical tool calls are blocked to prevent infinite loops
4. **Shell quoting fix** — common LLM quoting mistakes (e.g., `mkdir -p '{a,b}'`) are automatically corrected
5. **Context management** — at 50+ messages, tool results are truncated and project context is preserved
6. **Up to 40 iterations** — sufficient for complex multi-file projects

### Quality features

| Feature | What it does | Speed impact |
|---|---|---|
| **Chain-of-Thought prompting** | Models reason step-by-step for deeper, more accurate responses | Zero |
| **Streaming** (Groq) | Tokens arrive in real-time as the model generates | None (faster perceived speed) |
| **Self-reflection** | After display, Groq critiques and improves the response in background | None (async) |
| **Response cache** | Same (query, temperature) returns instantly on repeat | Positive |
| **Follow-up suggestions** | 2-3 relevant questions generated in background | None (async) |
| **Temperature auto-tune** | 0.2 factual, 0.4 code, 0.8 creative — auto-detected from query | Zero |
| **Response scoring** | Heuristic quality check discards low-quality responses | Zero |
| **Post-processing** | Strips filler phrases, normalizes formatting | Zero |

## Color reference

| Element | ANSI code | Sample |
|---|---|---|
| **Bold key terms** `**text**` | `\x1b[1;37m` (bold white) | **key concept** |
| Headings `### Title` | `\x1b[0;92m` (light green) | Section heading |
| Code blocks `` ``` `` | `\x1b[0;33m` (gold) | Multi-line code |
| Inline commands `` `cmd` `` | `\x1b[0;33m` (gold) | Commands, flags, paths |
| Shell prompts `$ cmd` | `\x1b[0;94m` (blue) | Command-line examples |
| Acronyms `API (...)` | `\x1b[33m` (gold) | Acronym definitions |
| Border | `\x1b[36m` (cyan) | Top/bottom horizontal rule |
| Table headers | `\x1b[1;93m` (bold yellow) | Table column headers |
| Table data | `\x1b[0;37m` (white) / `\x1b[0;90m` (gray) | Alternating row colors |

## Project structure

```
src/
  lib.rs    — core library: types, API calls, formatting, conversation,
              caching, streaming, self-reflection, follow-ups, project context
  main.rs   — binary entry point: CLI parsing, REPL loop with suggest command
  tools.rs  — coding agent tools: read, write, edit, bash (with quoting fix),
              grep, glob, list_dir
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

59 unit tests covering formatting, conversation, scoring, table rendering, and all 7 coding tools.

## License

[MIT](LICENSE)
