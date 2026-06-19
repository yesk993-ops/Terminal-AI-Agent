#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/yesk993-ops/Terminal-AI-Agent.git"
TARGET_DIR="$HOME/terminal-ai-agent"

GREEN='\033[0;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Terminal AI Agent — Quick Setup       ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# --------------------------------------------------
# Detect OS
# --------------------------------------------------
OS="$(uname -s)"
case "$OS" in
    Linux)
        # Detect distro
        if   [ -f /etc/debian_version ]; then DISTRO="debian"
        elif [ -f /etc/fedora-release ];  then DISTRO="fedora"
        elif [ -f /etc/arch-release ];    then DISTRO="arch"
        elif [ -f /etc/gentoo-release ];  then DISTRO="gentoo"
        elif [ -f /etc/SuSE-release ];    then DISTRO="suse"
        elif [ -f /etc/alpine-release ];  then DISTRO="alpine"
        else DISTRO="unknown"
        fi
        ;;
    Darwin)
        DISTRO="macos"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        DISTRO="windows"
        ;;
    *)
        echo -e "${YELLOW}Unsupported OS: $OS${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Detected: $OS ($DISTRO)${NC}"

# --------------------------------------------------
# Install system dependencies
# --------------------------------------------------
install_deps() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    case "$DISTRO" in
        debian)
            sudo apt-get update -qq
            sudo apt-get install -y -qq curl pkg-config libssl-dev build-essential git
            ;;
        fedora)
            sudo dnf install -y curl pkgconf-pkg-config openssl-devel gcc git
            ;;
        arch)
            sudo pacman -S --noconfirm curl pkg-config openssl base-devel git
            ;;
        gentoo)
            sudo emerge dev-libs/openssl dev-vcs/git
            ;;
        suse)
            sudo zypper install -y curl pkg-config libopenssl-devel gcc git
            ;;
        alpine)
            sudo apk add curl pkgconfig openssl-dev build-base git
            ;;
        macos)
            if ! command -v git &>/dev/null; then
                xcode-select --install 2>/dev/null || true
            fi
            ;;
        windows)
            echo -e "${YELLOW}On Windows, ensure you have:${NC}"
            echo "  - Git for Windows (https://git-scm.com)"
            echo "  - Rust from https://rustup.rs"
            echo "  - Node.js from https://nodejs.org (for OpenCode gateway)"
            echo "  - The Visual Studio C++ Build Tools (prompted by rustup)"
            echo ""
            echo "Then run this script again inside Git Bash or WSL."
            ;;
    esac
    echo -e "${GREEN}System dependencies OK.${NC}"
}

# --------------------------------------------------
# Install Rust via rustup if not present
# --------------------------------------------------
install_rust() {
    if command -v rustc &>/dev/null; then
        VER=$(rustc --version | cut -d' ' -f2)
        echo -e "${GREEN}Rust $VER already installed.${NC}"
    else
        echo -e "${YELLOW}Installing Rust via rustup...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        echo -e "${GREEN}Rust installed.${NC}"
    fi
}

# --------------------------------------------------
# Install Node.js if missing
# --------------------------------------------------
install_node() {
    if command -v node &>/dev/null; then
        NODE_VER=$(node --version)
        echo -e "${GREEN}Node.js $NODE_VER already installed.${NC}"
    else
        echo -e "${YELLOW}Installing Node.js...${NC}"
        case "$DISTRO" in
            debian)
                curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
                sudo apt-get install -y -qq nodejs
                ;;
            fedora)
                curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo -E bash -
                sudo dnf install -y nodejs
                ;;
            arch)
                sudo pacman -S --noconfirm nodejs npm
                ;;
            macos)
                if command -v brew &>/dev/null; then
                    brew install node
                else
                    curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-darwin-x64.tar.gz | sudo tar -xz -C /usr/local --strip-components=1
                fi
                ;;
            alpine)
                sudo apk add nodejs npm
                ;;
        esac
        echo -e "${GREEN}Node.js installed.${NC}"
    fi
}

# --------------------------------------------------
# Install OpenCode CLI + opencode-to-openai gateway
# --------------------------------------------------
install_opencode_gateway() {
    local GATEWAY_DIR="$HOME/.opencode-gateway"

    # Install OpenCode CLI via official script
    if ! command -v opencode &>/dev/null; then
        echo -e "${YELLOW}Installing OpenCode CLI...${NC}"
        curl -fsSL https://opencode.ai/install | bash 2>/dev/null || {
            echo -e "${YELLOW}Falling back to npm install...${NC}"
            npm install -g opencode-ai
        }
        echo -e "${GREEN}OpenCode CLI installed.${NC}"
    else
        echo -e "${GREEN}OpenCode CLI already installed.${NC}"
    fi

    # Clone / update gateway
    if [ -d "$GATEWAY_DIR" ]; then
        echo -e "${YELLOW}Updating opencode-to-openai gateway...${NC}"
        cd "$GATEWAY_DIR" && git pull --ff-only
    else
        echo -e "${YELLOW}Cloning opencode-to-openai gateway...${NC}"
        git clone --depth 1 https://github.com/dxxzst/opencode-to-openai.git "$GATEWAY_DIR"
        cd "$GATEWAY_DIR"
    fi

    cd "$GATEWAY_DIR"
    if [ ! -d node_modules ]; then
        echo -e "${YELLOW}Installing gateway dependencies...${NC}"
        npm install
    fi

    # Create a systemd user service (Linux) or launchd plist (macOS) for autostart
    case "$DISTRO" in
        debian|fedora|arch|gentoo|suse|alpine)
            local SERVICE_DIR="$HOME/.config/systemd/user"
            mkdir -p "$SERVICE_DIR"
            cat > "$SERVICE_DIR/opencode-gateway.service" << 'SERVICE'
[Unit]
Description=OpenCode-to-OpenAI API Gateway
After=network.target

[Service]
ExecStart=%h/.opencode-gateway/node index.js
WorkingDirectory=%h/.opencode-gateway
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICE
            systemctl --user daemon-reload 2>/dev/null || true
            systemctl --user enable --now opencode-gateway.service 2>/dev/null || true
            echo -e "${GREEN}Gateway systemd service installed and started.${NC}"
            ;;
        macos)
            mkdir -p "$HOME/Library/LaunchAgents"
            cat > "$HOME/Library/LaunchAgents/com.opencode-gateway.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.opencode-gateway</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/node</string>
        <string>index.js</string>
    </array>
    <key>WorkingDirectory</key>
    <string>%h/.opencode-gateway</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
PLIST
            launchctl load "$HOME/Library/LaunchAgents/com.opencode-gateway.plist" 2>/dev/null || true
            echo -e "${GREEN}Gateway launchd plist installed and started.${NC}"
            ;;
    esac

    # Also start immediately (in case service didn't, or on unsupported platforms)
    cd "$GATEWAY_DIR"
    nohup node index.js > /dev/null 2>&1 &
    sleep 1
    if curl -sf http://127.0.0.1:8083/health > /dev/null 2>&1; then
        echo -e "${GREEN}OpenCode gateway is running at http://127.0.0.1:8083${NC}"
    else
        echo -e "${YELLOW}Gateway may take a moment to start. Try: curl http://127.0.0.1:8083/health${NC}"
    fi
}

# --------------------------------------------------
# Detect target triple
# --------------------------------------------------
detect_target() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) echo -e "${YELLOW}Unknown arch: $arch${NC}"; return 1 ;;
    esac

    case "$OS" in
        Linux) echo "${arch}-unknown-linux-gnu" ;;
        Darwin)
            if [ "$arch" = "x86_64" ]; then
                echo "x86_64-apple-darwin"
            else
                echo "aarch64-apple-darwin"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "x86_64-pc-windows-gnu" ;;
        *) return 1 ;;
    esac
}

# --------------------------------------------------
# Download pre-built binary or fall back to source build
# --------------------------------------------------
download_or_build() {
    local target
    target="$(detect_target)" || true
    local BIN_NAME="terminal_ai_agent"
    local EXT=""
    case "$OS" in MINGW*|MSYS*|CYGWIN*) EXT=".exe" ;; esac

    # Try downloading pre-built binary first
    if [ -n "$target" ]; then
        local GH="https://github.com/yesk993-ops/Terminal-AI-Agent"
        local URL="$GH/releases/latest/download/${BIN_NAME}-${target}${EXT}"
        echo -e "${YELLOW}Downloading pre-built binary for ${target}...${NC}"
        if curl -fsL -o "$TARGET_DIR/${BIN_NAME}${EXT}" "$URL"; then
            chmod +x "$TARGET_DIR/${BIN_NAME}${EXT}"
            echo -e "${GREEN}Downloaded ${BIN_NAME} for ${target}.${NC}"
            return 0
        fi
        echo -e "${YELLOW}No pre-built binary available, building from source...${NC}"
    fi

    # Fall back to source build
    if [ ! -d "$TARGET_DIR" ]; then
        echo -e "${YELLOW}Cloning repository...${NC}"
        git clone --depth 1 "$REPO" "$TARGET_DIR"
    fi
    cd "$TARGET_DIR"

    echo -e "${YELLOW}Building release binary (this may take a while)...${NC}"
    cargo build --release
    echo -e "${GREEN}Build complete.${NC}"
}

# --------------------------------------------------
# Install globally (Linux/macOS)
# --------------------------------------------------
install_global() {
    # Find the binary (downloaded path vs built path)
    local BIN_SRC="$TARGET_DIR/terminal_ai_agent"
    if [ ! -f "$BIN_SRC" ]; then
        BIN_SRC="$TARGET_DIR/target/release/terminal_ai_agent"
    fi
    local EXT=""
    case "$OS" in MINGW*|MSYS*|CYGWIN*) EXT=".exe" ; BIN_SRC="${BIN_SRC}${EXT}" ;; esac

    if [ ! -f "$BIN_SRC" ]; then
        echo -e "${YELLOW}Binary not found at $BIN_SRC${NC}"
        return
    fi

    case "$DISTRO" in
        windows)
            echo -e "${YELLOW}Binary at: $BIN_SRC${NC}"
            echo -e "${YELLOW}Add it to your PATH manually, or copy:${NC}"
            echo "  copy \"$BIN_SRC\" \"C:\\Windows\\System32\\\""
            ;;
        *)
            echo -e "${YELLOW}Installing to /usr/local/bin/...${NC}"
            sudo cp "$BIN_SRC" /usr/local/bin/terminal_ai_agent$EXT
            sudo chmod +x /usr/local/bin/terminal_ai_agent$EXT
            # Install an `ask` wrapper so `ask <query>` works immediately
            sudo tee /usr/local/bin/ask > /dev/null << 'SCRIPT'
#!/usr/bin/env bash
exec /usr/local/bin/terminal_ai_agent "$@"
SCRIPT
            sudo chmod +x /usr/local/bin/ask
            echo -e "${GREEN}Installed to /usr/local/bin/terminal_ai_agent${NC}"
            echo -e "${GREEN}Installed ask wrapper to /usr/local/bin/ask${NC}"
            ;;
    esac
}

# --------------------------------------------------
# Optional: add ask() to shell rc for convenience in new terminals
# --------------------------------------------------
setup_rc_alias() {
    local rc
    case "$SHELL" in
        *zsh) rc="$HOME/.zshrc" ;;
        *bash) rc="$HOME/.bashrc" ;;
        *) rc="$HOME/.profile" ;;
    esac

    if grep -q "terminal_ai_agent" "$rc" 2>/dev/null; then
        return
    fi

    cat >> "$rc" << 'EOF'

# Terminal AI Agent — quick ask() shortcut
ask() {
    /usr/local/bin/terminal_ai_agent "$@"
}
EOF
    echo -e "${GREEN}Also added ask() to $rc for new terminals.${NC}"
}

# --------------------------------------------------
# Print next steps
# --------------------------------------------------
next_steps() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}  Setup complete!${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "Run: ${CYAN}ask 'your question'${NC}"
    echo ""
    echo -e "OpenCode gateway is running at ${CYAN}http://127.0.0.1:8083${NC}"
    echo -e "  → Free models available immediately: opencode/big-pickle, opencode/gpt-5-nano"
    echo ""

    if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${GROQ_API_KEY:-}" ]; then
        echo -e "${YELLOW}No API keys set — using OpenCode gateway + any env vars found.${NC}"
        echo ""
        echo -e "Optionally set keys for more providers:"
        echo "  export OPENROUTER_API_KEY=\"sk-or-v1-...\""
        echo "  export GROQ_API_KEY=\"gsk_...\""
        echo "  export GOOGLE_API_KEY=\"...\""
        echo "  export NVIDIA_API_KEY=\"nvapi-...\""
        echo ""
        echo -e "Add to ${CYAN}~/.bashrc${NC} (or ${CYAN}~/.zshrc${NC}) to persist."
        echo ""
        echo "Get free keys:"
        echo "  https://openrouter.ai/keys"
        echo "  https://console.groq.com/keys"
        echo "  https://aistudio.google.com/apikey"
        echo "  https://build.nvidia.com"
    fi
}

# --------------------------------------------------
# Main
# --------------------------------------------------
install_deps
install_rust
install_node
source "$HOME/.cargo/env" 2>/dev/null || true
download_or_build
install_global
install_opencode_gateway
setup_rc_alias
next_steps
