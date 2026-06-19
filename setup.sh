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
            # Xcode CLT are usually installed by rustup; just ensure git is present
            if ! command -v git &>/dev/null; then
                xcode-select --install 2>/dev/null || true
            fi
            ;;
        windows)
            echo -e "${YELLOW}On Windows, ensure you have:${NC}"
            echo "  - Git for Windows (https://git-scm.com)"
            echo "  - Rust from https://rustup.rs"
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

    echo -e "${YELLOW}Building release binary...${NC}"
    cargo build --release
    echo -e "${GREEN}Build complete.${NC}"
}

# --------------------------------------------------
# Install globally (Linux/macOS)
# --------------------------------------------------
install_global() {
    case "$DISTRO" in
        windows)
            echo -e "${YELLOW}Binary at: $TARGET_DIR/target/release/terminal_ai_agent.exe${NC}"
            echo -e "${YELLOW}Add it to your PATH manually, or copy:${NC}"
            echo "  copy \"$TARGET_DIR\\target\\release\\terminal_ai_agent.exe\" \"C:\\Windows\\System32\\\""
            ;;
        *)
            echo -e "${YELLOW}Installing to /usr/local/bin/...${NC}"
            sudo cp "$TARGET_DIR/target/release/terminal_ai_agent" /usr/local/bin/
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

    if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${GROQ_API_KEY:-}" ]; then
        echo -e "${YELLOW}Next: set at least one API key:${NC}"
        echo ""
        echo "  export OPENROUTER_API_KEY=\"sk-or-v1-...\""
        echo "  # or"
        echo "  export GROQ_API_KEY=\"gsk_...\""
        echo ""
        echo -e "Add the line above to ${CYAN}~/.bashrc${NC} (or ${CYAN}~/.zshrc${NC}) to persist."
        echo ""
        echo "Get a free key at:"
        echo "  https://openrouter.ai/keys"
        echo "  https://console.groq.com/keys"
    fi
}

# --------------------------------------------------
# Main
# --------------------------------------------------
install_deps
install_rust
source "$HOME/.cargo/env" 2>/dev/null || true
build_project
install_global
setup_rc_alias
next_steps
