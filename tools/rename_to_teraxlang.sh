#!/bin/bash
# Script to rename triton -> teraxlang_triton and txl -> teraxlang
# Run this once on a fresh checkout of thirdparty/triton

set -e

TRITON_DIR="thirdparty/triton/python"

cd "$(dirname "$0")/.."

echo "=== Renaming packages for TeraXLang ==="

# Rename triton to teraxlang_triton
if [ -d "$TRITON_DIR/triton" ]; then
    echo "Renaming triton -> teraxlang_triton..."
    mv "$TRITON_DIR/triton" "$TRITON_DIR/teraxlang_triton"
fi

# Rename txl to teraxlang
if [ -d "$TRITON_DIR/txl" ]; then
    echo "Renaming txl -> teraxlang..."
    mv "$TRITON_DIR/txl" "$TRITON_DIR/teraxlang"
fi

# Replace imports in teraxlang_triton (triton -> teraxlang_triton)
if [ -d "$TRITON_DIR/teraxlang_triton" ]; then
    echo "Replacing imports in teraxlang_triton..."
    python3 tools/replace_imports.py "$TRITON_DIR/teraxlang_triton" triton teraxlang_triton
fi

# Replace imports in teraxlang
if [ -d "$TRITON_DIR/teraxlang" ]; then
    echo "Replacing imports in teraxlang (triton -> teraxlang_triton)..."
    python3 tools/replace_imports.py "$TRITON_DIR/teraxlang" triton teraxlang_triton
    
    echo "Replacing imports in teraxlang (txl -> teraxlang)..."
    python3 tools/replace_imports.py "$TRITON_DIR/teraxlang" txl teraxlang
fi

echo "=== Done ==="
echo "Directories created:"
ls -d "$TRITON_DIR"/teraxlang* 2>/dev/null || echo "  (none found)"
