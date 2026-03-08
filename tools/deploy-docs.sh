#!/bin/bash
# Deploy TXL documentation to GitHub Pages

set -e

export PATH="$HOME/Library/Python/3.12/bin:$PATH"

echo "Installing mkdocs and plugins..."
python3 -m pip install --user --break-system-packages mkdocs mkdocs-material mkdocstrings mkdocstrings-python

echo "Building documentation..."
mkdocs build

echo "Deploying to GitHub Pages..."
mkdocs gh-deploy
