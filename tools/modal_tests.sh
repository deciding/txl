#!/bin/bash
# Modal test runner - runs Modal tests with logging and downloads dump files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Check if script name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <modal_script.py> [test_name] [volume_name] [debug_env]"
    echo ""
    echo "Arguments:"
    echo "  modal_script.py  Name of Modal test script in docker/ directory"
    echo "  test_name       Optional custom test name for dump directory"
    echo "  volume_name     Optional Modal volume name (default: txl-dump)"
    echo "  debug_env       Optional debug env var (e.g., TXLGPU_DEBUG=txlgpu-pipeline)"
    echo ""
    echo "Examples:"
    echo "  $0 flash_attention.py"
    echo "  $0 mla_decoding.py my-test"
    echo "  $0 flash_attention.py my-test txl-dump"
    echo "  $0 nsa_prefill.py my-test txl-dump TXLGPU_DEBUG=txlgpu-pipeline"
    exit 1
fi

# Default volume name
VOLUME_NAME="${3:-txl-dump}"

# Debug environment variable (4th argument)
DEBUG_ENV="${4:-}"

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

# Set debug env var if provided
if [ -n "$DEBUG_ENV" ]; then
    export $DEBUG_ENV
fi

DUMP_DIR="$DUMP_DIR_NAME" modal run "$1" > "$LOG_FILE" 2>&1

echo ""
echo "... [full log in $LOG_FILE]"
echo ""

# Show last 200 lines (results/errors) after completion
tail -200 "$LOG_FILE"

echo ""
echo "=== Downloading dump files from Modal volume ==="

# Download only the specific dump folder from Modal volume
# Download to parent directory to avoid nested structure
# Use --force to overwrite if local directory already exists
echo "Downloading: $VOLUME_NAME/$DUMP_DIR_NAME -> $DUMPS_ROOT/"
modal volume get "$VOLUME_NAME" "$DUMP_DIR_NAME" "$DUMPS_ROOT/" --force 2>/dev/null || {
    echo "Warning: Could not download dump files from volume"
    echo "Volume contents:"
    modal volume ls "$VOLUME_NAME" 2>/dev/null || echo "Volume not found"
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