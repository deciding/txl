#!/bin/bash
# GCP Teardown Script - Stop or delete VM and persistent resources
# Usage: ./gcp-teardown.sh [--keep-disk] [--delete-all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp-config.sh"

KEEP_DISK=false
DELETE_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-disk)
            KEEP_DISK=true
            shift
            ;;
        --delete-all)
            DELETE_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== GCP TXL Teardown ==="

# Stop the VM
echo "Stopping VM..."
gcloud compute instances stop "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" 2>/dev/null || true

if [[ "$DELETE_ALL" == "true" ]]; then
    echo "Deleting VM..."
    gcloud compute instances delete "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" --quiet 2>/dev/null || true
    
    if [[ "$KEEP_DISK" == "false" ]]; then
        echo "Deleting disk..."
        gcloud compute disks delete "${VM_NAME}-data" --zone="$GCP_ZONE" --project="$GCP_PROJECT" --quiet 2>/dev/null || true
    fi
fi

echo ""
echo "=== Teardown Complete ==="
echo ""
if [[ "$KEEP_DISK" == "true" ]]; then
    echo "VM stopped. Disk preserved for next run."
    echo "Cost: ~$5-10/month for persistent disk"
else
    echo "VM and disk deleted."
    echo "Only GCS bucket data remains."
fi

echo ""
echo "To restart:"
echo "  ./gcp-provision.sh --start-existing"
