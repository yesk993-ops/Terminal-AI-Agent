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
# Clone / update repo and build
# --------------------------------------------------
build_project() {
    if [ -d "$TARGET_DIR" ]; then
        echo -e "${YELLOW}Updating existing clone...${NC}"
        cd "$TARGET_DIR" && git pull --ff-only
    else
        echo -e "${YELLOW}Cloning repository...${NC}"
        git clone --depth 1 "$REPO" "$TARGET_DIR"
        cd "$TARGET_DIR"
    fi

    echo -e "${YELLOW}Building release binary (this may take a while)...${NC}"
    cargo build --release
    echo -e "${GREEN}Build complete.${NC}"
}

# --------------------------------------------------
# Install globally (Linux/macOS)
# --------------------------------------------------
install_global() {
    case "$DISTRO" in
        windows)
            echo -e "${YELLOW}Installing globally to C:\\Windows\\System32\\...${NC}"
            # Convert Unix path to Windows path for the binary
            local BIN_PATH="$TARGET_DIR/target/release/terminal_ai_agent.exe"
            if [ -f "$BIN_PATH" ]; then
                # Try to auto-install (requires admin) — fall back gracefully
                # rm before cp to avoid "Text file busy" when binary is running
                rm -f "C:/Windows/System32/terminal_ai_agent.exe" 2>/dev/null || true
                if cp "$BIN_PATH" "C:/Windows/System32/terminal_ai_agent.exe" 2>/dev/null; then
                    chmod +x "C:/Windows/System32/terminal_ai_agent.exe" 2>/dev/null || true
                    # Create ask.cmd wrapper so 'ask' works from cmd/powershell too
                    cat > "C:/Windows/System32/ask.cmd" << 'BAT'
@echo off
terminal_ai_agent %*
BAT
                    echo -e "${GREEN}Installed terminal_ai_agent.exe to C:\\Windows\\System32\\${NC}"
                    echo -e "${GREEN}Installed ask.cmd wrapper — use 'ask' from cmd, PowerShell, or Git Bash${NC}"
                else
                    echo -e "${YELLOW}Could not write to C:\\Windows\\System32\\(run as Admin?).${NC}"
                    echo -e "${YELLOW}Binary at: $BIN_PATH${NC}"
                    echo -e "${YELLOW}To install manually, run this terminal as Administrator and run:${NC}"
                    echo "  copy \"$(cygpath -w "$BIN_PATH" 2>/dev/null || echo "$BIN_PATH")\" \"C:\Windows\System32\""
                fi
            else
                echo -e "${YELLOW}Binary not found at $BIN_PATH — skipping global install.${NC}"
            fi
            ;;
        *)
            echo -e "${YELLOW}Installing to /usr/local/bin/...${NC}"
            # rm before cp to avoid "Text file busy" when the binary is currently running
            sudo rm -f /usr/local/bin/terminal_ai_agent /usr/local/bin/ask
            sudo cp "$TARGET_DIR/target/release/terminal_ai_agent" /usr/local/bin/
            sudo chmod +x /usr/local/bin/terminal_ai_agent
            # Install an `ask` wrapper so `ask <query>` works immediately
            printf '#!/usr/bin/env bash\nexec /usr/local/bin/terminal_ai_agent "$@"\n' | sudo tee /usr/local/bin/ask > /dev/null
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

    # Add ask() function if not already present
    if ! grep -q "terminal_ai_agent" "$rc" 2>/dev/null; then
        cat >> "$rc" << 'EOF'

# Terminal AI Agent — quick ask() shortcut
ask() {
    /usr/local/bin/terminal_ai_agent "$@"
}
EOF
        echo -e "${GREEN}Added ask() to $rc for new terminals.${NC}"
    fi

    # Persist NVIDIA_API_KEY in shell rc if currently set (primary provider, no rate limits)
    if [ -n "${NVIDIA_API_KEY:-}" ] && ! grep -q "NVIDIA_API_KEY" "$rc" 2>/dev/null; then
        cat >> "$rc" << EOF

# NVIDIA NIM API key (primary provider — production models, no rate limits)
export NVIDIA_API_KEY="${NVIDIA_API_KEY}"
EOF
        echo -e "${GREEN}Persisted NVIDIA_API_KEY to $rc.${NC}"
    fi
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

    echo -e "${YELLOW}Recommended: Set your NVIDIA API key (production models, no rate limits)${NC}"
    echo ""
    echo -e "  Get a key: ${CYAN}https://build.nvidia.com${NC}"
    echo ""
    echo -e "  ${YELLOW}export NVIDIA_API_KEY=\"nvapi-...\"${NC}"
}

# --------------------------------------------------
# Main
# --------------------------------------------------
install_deps
install_rust
install_node
source "$HOME/.cargo/env" 2>/dev/null || true
build_project
install_global
install_opencode_gateway
setup_rc_alias
next_steps
