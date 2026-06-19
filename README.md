# Terminal AI Agent

A fast, colorful terminal-based AI chatbot in Rust. Queries free models from
**OpenRouter** and **Groq** in parallel — the fastest response wins.

## Features

- Parallel model racing — tries OpenRouter + Groq simultaneously, <300ms typical
- Conversation memory (last 6 turns, persisted across restarts)
- Color-coded output:
  - **Bold terms** → bold cyan
  - **Headings** (`###`) → bold yellow
  - **Code blocks** → amber
  - **Inline commands** `` `cmd` `` → green
  - **Shell command lines** (`$ ...`) → amber
- Top/bottom border only — copy-paste friendly
- REPL mode (`ask <question>`) or single-shot CLI
- Ctrl+C graceful shutdown with conversation save
- Configurable temperature via `--temp`

## Prerequisites

- Rust toolchain 1.80+ ([rustup](https://rustup.rs/))
- At least one API key:
  ```bash
  export OPENROUTER_API_KEY="sk-or-..."
  export GROQ_API_KEY="gsk_..."           # optional, used as fallback
  ```

## Build

```bash
cargo build --release
```

## Run

```bash
# Single query
./target/release/terminal_ai_agent "What is Rust?"

# With temperature override
./target/release/terminal_ai_agent --temp 0.7 "Explain monads"

# REPL mode (type 'exit' to quit)
./target/release/terminal_ai_agent
  > ask How does AWS work?
```

## Tests

```bash
cargo test
```

## Project structure

```
src/
  lib.rs    — library crate: types, API calls, formatting, conversation, tests
  main.rs   — binary entry point: CLI parsing, REPL loop
```

## License

MIT
