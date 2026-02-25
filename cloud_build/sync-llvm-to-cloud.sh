#!/bin/bash
# Sync LLVM to GCS - Upload pre-downloaded LLVM to cloud storage
# Usage: ./sync-llvm-to-cloud.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp-config.sh"

echo "=== Syncing LLVM to GCS ==="

LLVM_DIR="${PROJECT_ROOT}/llvm-7d5de303-ubuntu-x64"
LLVM_TAR="${PROJECT_ROOT}/llvm-7d5de303-ubuntu-x64.tar.gz"

# Check if LLVM exists
if [ ! -d "$LLVM_DIR" ]; then
    echo "Error: LLVM directory not found at $LLVM_DIR"
    exit 1
fi

# Compress if tar doesn't exist
if [ ! -f "$LLVM_TAR" ]; then
    echo "Compressing LLVM..."
    tar -czf "$LLVM_TAR" -C "$PROJECT_ROOT" "llvm-7d5de303-ubuntu-x64"
fi

# Upload to GCS
echo "Uploading to gs://${GCS_BUCKET_NAME}-llvm/"
gsutil cp "$LLVM_TAR" "gs://${GCS_BUCKET_NAME}-llvm/"

echo ""
echo "=== LLVM Synced ==="
echo "Location: gs://${GCS_BUCKET_NAME}-llvm/llvm-7d5de303-ubuntu-x64.tar.gz"
echo ""
echo "On VM, download with:"
echo "  gsutil cp gs://${GCS_BUCKET_NAME}-llvm/llvm-7d5de303-ubuntu-x64.tar.gz ."
