#!/bin/bash

set -e

# ==============================
# FedBank Repository Clone Script
# ==============================

REPO_URL="https://github.com/ABHINC03/FEDBANK.git"
PROJECT_DIR="$HOME/FedBank"

echo "================================"
echo " FedBank Repository Setup"
echo "================================"

# Check Git
if ! command -v git &> /dev/null; then
    echo "Git is not installed."
    echo "Installing Git..."
    sudo apt update
    sudo apt install -y git
fi

# Clone repository
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Repository already exists."
    echo "Updating repository..."

    cd "$PROJECT_DIR"
    git pull

else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

echo ""
echo "================================"
echo " Repository Ready!"
echo "================================"
echo ""
echo "Location: $PROJECT_DIR"
echo ""
echo "To enter the project:"
echo "cd $PROJECT_DIR"