#!/bin/bash
# GCP Configuration for TXL Cloud Build
# Edit these values for your setup

# Project settings
export GCP_PROJECT="${GCP_PROJECT:-txl-build}"
export GCP_REGION="${GCP_REGION:-us-central1}"
export GCP_ZONE="${GCP_ZONE:-us-central1-a}"

# Storage bucket names (must be globally unique)
export GCS_BUCKET_NAME="${GCS_BUCKET_NAME:-txl-build-cache}"
export GCS_LOCATION="${GCS_LOCATION:-US}"

# VM settings
export VM_NAME="${VM_NAME:-txl-builder}"
export VM_MACHINE_TYPE="${VM_MACHINE_TYPE:-e2-standard-8}"  # 8 vCPU, 32GB RAM
export VM_DISK_SIZE="${VM_DISK_SIZE:-100GB}"
export VM_BOOT_DISK_SIZE="${VM_BOOT_DISK_SIZE:-50GB}"

# Build settings
export MAX_JOBS="${MAX_JOBS:-8}"
export TRITON_BUILD_WITH_O1="${TRITON_BUILD_WITH_O1:-1}"
export TRITON_BUILD_PROTON="${TRITON_BUILD_PROTON:-0}"
export TRITON_BUILD_WITH_CLANG_LLD="${TRITON_BUILD_WITH_CLANG_LLD:-0}"

# Local paths (relative to project root)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${PROJECT_ROOT}/output"
export CLOUD_BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Service account (optional, if not set will use default)
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"

# Pre-mount path on VM
export VM_MOUNT_POINT="/mnt/txl-cache"
