#!/bin/bash
echo "Checking for Dioxus CLI (dx)..."
if ! command -v dx &> /dev/null; then
    echo "Dioxus CLI not found. Setting up Rust toolchain and dependencies..."
    
    echo "Installing Rust stable toolchain..."
    rustup toolchain install stable
    
    echo "Adding wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
    
    echo "Installing cargo-binstall..."
    cargo install cargo-binstall
    
    echo "Installing dioxus-cli using cargo-binstall..."
    cargo binstall dioxus-cli --no-confirm
    
    echo "Dioxus CLI installed successfully."
else
    echo "Dioxus CLI is already installed."
fi

echo "Starting Queues development server..."
dx serve --platform web --addr 0.0.0.0 --port 8082 --fullstack
