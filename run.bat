@echo off
echo Checking for Dioxus CLI (dx)...
dx --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Dioxus CLI not found. Setting up Rust toolchain and dependencies...
    
    echo Installing Rust stable toolchain...
    rustup toolchain install stable
    
    echo Adding wasm32-unknown-unknown target...
    rustup target add wasm32-unknown-unknown
    
    echo Installing cargo-binstall...
    cargo install cargo-binstall
    
    echo Installing dioxus-cli using cargo-binstall...
    cargo binstall dioxus-cli --no-confirm
    
    echo Dioxus CLI installed successfully.
) else (
    echo Dioxus CLI is already installed.
)

echo Starting Queues development server...
dx serve --platform web --addr 127.0.0.1 --port 8082 --fullstack
