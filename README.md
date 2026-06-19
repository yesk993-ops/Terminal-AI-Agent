<div align="center">

# Terminal AI Agent

**A fast, colorful AI agent for your terminal — powered by free models from OpenRouter and Groq.**

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-dea584?logo=rust)](https://www.rust-lang.org)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-yesk993--ops/Terminal--AI--Agent-181717?logo=github)](https://github.com/yesk993-ops/Terminal-AI-Agent)

```text
──────────────────────────────────────────────────────────────────────────────
Auto Scaling in AWS is a web service that allows you to scale your resources
up or down automatically based on predefined conditions. It's designed to
ensure that your application has the necessary resources to handle changes
in load, while also minimizing costs.

Key Components:

  Launch Configuration — template defining AMI, instance type, security groups
  Auto Scaling Group  — manages scaling of resources
  Scaling Policies    — rules that determine when to launch or terminate
──────────────────────────────────────────────────────────────────────────────
```

</div>

## Features

- ⚡ **Parallel model racing** — queries OpenRouter and Groq simultaneously; fastest response wins (<300ms typical)
- 🎨 **Color-coded output** — bold terms, headings, code blocks, inline commands each have distinct eye-friendly colors
- 📦 **Zero config** — set one API key and go
- 💬 **Conversation memory** — remembers last 6 turns, persisted across restarts
- 📋 **Copy-friendly** — top/bottom border only, no side walls
- 🔄 **REPL & single-shot** — interactive mode with `ask` or one-liner from shell
- 🧵 **Graceful shutdown** — Ctrl+C saves conversation before exit

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/yesk993-ops/Terminal-AI-Agent/main/setup.sh | bash
```

The script auto-detects your OS, installs Rust + dependencies, builds the project, and installs the binary. After it finishes, set your API key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
terminal_ai_agent "What is Rust?"
```

> 📖 For detailed platform-specific instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Single query
```bash
terminal_ai_agent "Explain monads in functional programming"
```

### Custom temperature
```bash
terminal_ai_agent --temp 0.7 "Write a haiku about Rust"
```

### REPL mode
```bash
terminal_ai_agent
> ask What is AWS Auto Scaling?
> ask How do I set up an S3 bucket?
> exit
```

### Shell shortcut
Add to `~/.bashrc`:
```bash
ask() { terminal_ai_agent "$@"; }
```
Then:
```bash
ask What is the capital of France?
```

## How it works

```
┌──────────┐     ┌──────────────────┐     ┌─────────┐
│  Query   │ ──▶ │  Parallel race   │ ──▶ │  Fastest │
│          │     │                  │     │  answer  │
│  user    │     │  OpenRouter 🤖   │     │   wins   │
│  types   │     │  Groq       ⚡    │     │          │
└──────────┘     └──────────────────┘     └─────────┘
```

The agent starts all models from both providers simultaneously. Each has a 10-second timeout. The first model to return a valid response wins — so even if OpenRouter models are slow, Groq answers in under 300ms.

## Color reference

| Element | Color | Purpose |
|---|---|---|
| **Bold text** `**term**` | Bold cyan | Key concepts |
| Headings `### Title` | Bold yellow | Section breaks |
| Code blocks ` ``` ` | Amber | Scripts & multi-line code |
| Inline commands `` `cmd` `` | Green | Commands, flags, paths |
| Shell prompts `$ cmd` | Amber | Command-line examples |

## API keys

| Provider | Env var | Where to get it |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| Groq | `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |

Keys are **never** stored in source code — only read from environment variables at runtime.

## Project structure

```
src/
  lib.rs    — core library: types, API calls, formatting, conversation, tests
  main.rs   — binary entry point: CLI parsing, REPL loop
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
