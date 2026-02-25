#!/bin/bash
# GCP Build Script - Run build on cloud VM
# Usage: ./gcp-build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp-config.sh"

echo "=== GCP TXL Build ==="

# Check if VM is running
VM_STATUS=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$VM_STATUS" != "RUNNING" ]]; then
    echo "VM is not running. Starting..."
    gcloud compute instances start "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT"
    
    echo "Waiting for VM to start..."
    for i in {1..30}; do
        VM_STATUS=$(gcloud compute instances describe "$VM_NAME" \
            --zone="$GCP_ZONE" \
            --project="$GCP_PROJECT" \
            --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
        if [[ "$VM_STATUS" == "RUNNING" ]]; then
            break
        fi
        sleep 10
    done
fi

VM_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo "VM IP: $VM_IP"

# Sync source code (only triton folder + build scripts)
echo ""
echo "=== Syncing source code ==="
gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${PROJECT_ROOT}/tools/build-wheel.sh" \
    "${PROJECT_ROOT}/tools/build-wheel-docker.sh" \
    "${PROJECT_ROOT}/tools/Dockerfile.txl-wheel" \
    "${PROJECT_ROOT}/thirdparty/triton" \
    "${PROJECT_ROOT}/llvm-7d5de303-ubuntu-x64" \
    "${VM_NAME}:/txl/" 2>/dev/null || true

# If previous sync failed, try individual items
echo "Syncing to VM..."
gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${PROJECT_ROOT}/tools/build-wheel.sh" \
    "${VM_NAME}:/home/"

gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${PROJECT_ROOT}/tools/build-wheel-docker.sh" \
    "${VM_NAME}:/home/"

gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${PROJECT_ROOT}/tools/Dockerfile.txl-wheel" \
    "${VM_NAME}:/home/"

gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${PROJECT_ROOT}/thirdparty/triton" \
    "${VM_NAME}:/home/"

gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${PROJECT_ROOT}/llvm-7d5de303-ubuntu-x64" \
    "${VM_NAME}:/home/" || echo "Note: LLVM not synced, will use cloud storage"

# Create output directory on VM
gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" -- "mkdir -p /home/output"

# Run build on VM
echo ""
echo "=== Running build on VM ==="
gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" -- << 'BUILDSCRIPT'
set -e

cd /home

# Check if LLVM exists
if [ -d "llvm-7d5de303-ubuntu-x64" ]; then
    echo "Using local LLVM"
    export LLVM_SYSPATH=/home/llvm-7d5de303-ubuntu-x64
else
    echo "LLVM not found locally"
fi

# Build the wheel
export MAX_JOBS=${MAX_JOBS:-8}
export TRITON_BUILD_WITH_O1=${TRITON_BUILD_WITH_O1:-1}
export TRITON_BUILD_PROTON=${TRITON_BUILD_PROTON:-0}

# Use local build instead of docker (faster for repeated builds)
cd thirdparty/triton

# Set compiler paths
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export PATH=/opt/conda/bin:/opt/conda/envs/txl/bin:$PATH

# Check if conda environment exists
if ! conda env list | grep -q txl; then
    echo "Creating conda environment..."
    conda create -n txl python=3.12 -y
fi

conda activate txl

# Install dependencies if not present
pip install torch numpy matplotlib pandas

# Run build
echo "Building with MAX_JOBS=$MAX_JOBS"
python setup.py bdist_wheel

# Copy wheel to output
mkdir -p /home/output
cp dist/*.whl /home/output/

echo "Build complete!"
ls -lh /home/output/
BUILDSCRIPT

# Copy wheel back to local
echo ""
echo "=== Copying wheel back ==="
mkdir -p "${OUTPUT_DIR}"
gcloud compute scp --recurse \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    "${VM_NAME}:/home/output/*" \
    "${OUTPUT_DIR}/"

echo ""
echo "=== Build Complete ==="
ls -lh "${OUTPUT_DIR}/"
