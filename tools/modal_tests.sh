#!/bin/bash
# Modal test runner - runs Modal tests with logging and downloads dump files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Check if script name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <modal_script.py> [test_name]"
    echo ""
    echo "Arguments:"
    echo "  modal_script.py  Name of Modal test script in docker/ directory"
    echo "  test_name       Optional custom test name for dump directory"
    echo ""
    echo "Examples:"
    echo "  $0 flash_attention.py"
    echo "  $0 matmul.py my-custom-test"
    exit 1
fi

MODAL_SCRIPT="$DOCKER_DIR/$1"

# Check if modal script exists
if [ ! -f "$MODAL_SCRIPT" ]; then
    echo "Error: Modal script not found: $MODAL_SCRIPT"
    exit 1
fi

# Create dump directory with same name as log file (without .log extension)
TEST_NAME="${2:-$1}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DUMP_DIR_NAME="${TEST_NAME%.py}_${TIMESTAMP}"
DUMPS_ROOT="$DOCKER_DIR/dumps"
DUMP_DIR="$DUMPS_ROOT/$DUMP_DIR_NAME"
LOG_FILE="$DUMP_DIR.log"

mkdir -p "$DUMPS_ROOT"

echo "=== Modal Test Runner ==="
echo "Script: $1"
echo "Dump directory name: $DUMP_DIR_NAME"
echo "Local dump path: $DUMP_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Run Modal with DUMP_DIR env var and output redirected to log file
cd "$DOCKER_DIR"
DUMP_DIR="$DUMP_DIR_NAME" modal run "$1" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Downloading dump files from Modal volume ==="

# Download only the specific dump folder from Modal volume
# Download to parent directory to avoid nested structure
echo "Downloading: txl-dump/$DUMP_DIR_NAME -> $DUMPS_ROOT/"
modal volume get txl-dump "$DUMP_DIR_NAME" "$DUMPS_ROOT/" 2>/dev/null || {
    echo "Warning: Could not download dump files from volume"
    echo "Volume contents:"
    modal volume ls txl-dump 2>/dev/null || echo "Volume not found"
}

echo ""
echo "=== Test completed ==="
echo "Output saved to: $LOG_FILE"
echo "Dump files saved to: $DUMP_DIR"
if [ -d "$DUMP_DIR" ]; then
    echo ""
    echo "Dump contents:"
    ls -la "$DUMP_DIR"
fi