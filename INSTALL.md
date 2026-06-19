# Installation Guide

## Quick setup (all platforms, one command)

**Linux / macOS**
```bash
curl -fsSL https://raw.githubusercontent.com/yesk993-ops/Terminal-AI-Agent/main/setup.sh | bash
```

**Windows** — run inside [Git Bash](https://git-scm.com) or WSL:
```bash
curl -fsSL https://raw.githubusercontent.com/yesk993-ops/Terminal-AI-Agent/main/setup.sh | bash
```

The script will:
1. Detect your OS (Debian/Ubuntu, Fedora, Arch, macOS, Windows)
2. Install system dependencies (libssl, pkg-config, build tools)
3. Install Rust via rustup (if not already present)
4. Clone and build the project
5. Install the binary to `/usr/local/bin/` (Linux/macOS)
6. Print API key setup instructions

After it finishes, set your API key and you're done:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
terminal_ai_agent "What is Rust?"
```

---

## Manual installation

### Prerequisites

#### Step 1: Install Rust

**Linux (any distro)**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**macOS**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```
Or via [Homebrew](https://brew.sh/):
```bash
brew install rust
```

**Windows**
1. Download the installer from [rustup.rs](https://rustup.rs/) and run it.
2. During install, choose **"Default host triple"** (usually `x86_64-pc-windows-msvc`).
3. If prompted, also install the **Visual Studio Build Tools** or **Visual Studio C++ Build Tools** (required by the Rust compiler on Windows).
4. After installation, open a **new** Command Prompt or PowerShell window.

Verify the installation on any platform:
```bash
rustc --version   # Should print rustc 1.80 or later (the project requires 1.80+)
cargo --version
```

### Step 2: System dependencies

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install -y pkg-config libssl-dev build-essential
```

**Linux (Fedora/RHEL)**
```bash
sudo dnf install openssl-devel pkgconf-pkg-config gcc
```

**Linux (Arch)**
```bash
sudo pacman -S openssl pkg-config base-devel
```

**macOS**
No extra system packages needed — Xcode Command Line Tools are pulled in by Rust automatically if missing.

**Windows**
No extra system packages needed — the Rust installer handles everything.

---

## Installation

### Clone the repository
```bash
git clone https://github.com/yesk993-ops/Terminal-AI-Agent.git
cd Terminal-AI-Agent
```

### Build (all platforms)
```bash
cargo build --release
```

The binary will be at `./target/release/terminal_ai_agent`.

### (Optional) Install globally

**Linux / macOS**
```bash
sudo cp target/release/terminal_ai_agent /usr/local/bin/
```

**Windows (PowerShell as Administrator)**
```powershell
Copy-Item .\target\release\terminal_ai_agent.exe C:\Windows\System32\
```

---

## API Keys (no hardcoded secrets)

This agent **never** stores API keys in source code. All keys are read from
environment variables at runtime.

### OpenRouter (recommended primary)
1. Sign up at [openrouter.ai](https://openrouter.ai/keys).
2. Create an API key.
3. Export it in your shell profile:

**Linux / macOS** (`~/.bashrc`, `~/.zshrc`, or `~/.profile`)
```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxx"
```

**Windows PowerShell** (`$PROFILE`)
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxx"
# Persist for future sessions:
[Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-v1-xxxxxxxxxxxxxxxx", "User")
```

### Groq (optional fallback — faster, free)
1. Sign up at [console.groq.com/keys](https://console.groq.com/keys).
2. Create an API key.
3. Export it:

```bash
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxx"
```

The agent tries OpenRouter first, then Groq in parallel. The fastest response
wins. If only `GROQ_API_KEY` is set (no `OPENROUTER_API_KEY`), it falls back
to Groq automatically.

---

## Usage

### Single query
```bash
terminal_ai_agent "What is Rust?"
```

### With custom temperature
```bash
terminal_ai_agent --temp 0.7 "Explain monads"
```

### REPL mode
```bash
terminal_ai_agent
> ask How does AWS Auto Scaling work?
> ask What about EC2?
> exit
```

### Shell function (convenient `ask` command)
Add to `~/.bashrc` / `~/.zshrc`:
```bash
ask() {
    /path/to/terminal_ai_agent "$@"
}
```
Then use:
```bash
ask What is the capital of France?
```

---

## Uninstall

**Remove the binary:**
```bash
rm /usr/local/bin/terminal_ai_agent               # Linux / macOS
del C:\Windows\System32\terminal_ai_agent.exe      # Windows
```

**Remove persistent conversation history:**
```bash
rm -rf ~/.local/share/terminal_ai_agent
```

**Remove the cloned repo:**
```bash
rm -rf Terminal-AI-Agent
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `OPENROUTER_API_KEY not set` | Export the env var (see API Keys section above) |
| `All free models failed. Timeout` | Check your internet or try with `--temp 0.5` |
| `HTTP 401 (Unauthorized)` | Your API key is invalid — regenerate it |
| `HTTP 429 (Rate limited)` | Wait a minute and try again |
| `openssl-sys build failed` | Install `libssl-dev` (Linux) or Xcode CLT (macOS) |
| No colored output | Ensure your terminal supports ANSI escape codes (most do) |
