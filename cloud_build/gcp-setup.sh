#!/bin/bash
# GCP Setup Script - Run once to initialize GCP resources
# Usage: ./gcp-setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp-config.sh"

echo "=== GCP TXL Cloud Build Setup ==="
echo "Project: $GCP_PROJECT"
echo "Region: $GCP_REGION"
echo "Bucket: $GCS_BUCKET_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if logged in
echo "=== Checking GCP authentication ==="
gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || {
    echo "Not logged in. Running gcloud auth login..."
    gcloud auth login
}

# Set project
echo ""
echo "=== Setting project ==="
gcloud config set project "$GCP_PROJECT" 2>/dev/null || {
    echo "Project doesn't exist. Creating..."
    gcloud projects create "$GCP_PROJECT" --name="TXL Build"
    gcloud config set project "$GCP_PROJECT"
}

# Enable required APIs
echo ""
echo "=== Enabling required APIs ==="
echo "This may take a few minutes..."
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    --project="$GCP_PROJECT" \
    --quiet

# Create GCS bucket
echo ""
echo "=== Creating GCS bucket ==="
if gsutil ls "gs://$GCS_BUCKET_NAME/" &> /dev/null; then
    echo "Bucket $GCS_BUCKET_NAME already exists"
else
    gsutil mb -l "$GCS_LOCATION" "gs://$GCS_BUCKET_NAME/"
    echo "Created bucket: gs://$GCS_BUCKET_NAME/"
fi

# Set lifecycle policy to keep costs low (delete objects older than 30 days)
echo ""
echo "=== Setting lifecycle policy ==="
cat > /tmp/lifecycle-policy.json << 'EOF'
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
EOF
gsutil lifecycle set /tmp/lifecycle-policy.json "gs://$GCS_BUCKET_NAME/"
rm /tmp/lifecycle-policy.json

# Create directory structure in bucket
echo ""
echo "=== Creating bucket structure ==="
gsutil ls "gs://$GCS_BUCKET_NAME/" 2>/dev/null || true
gsutil mb -l "$GCS_LOCATION" "gs://$GCS_BUCKET_NAME-build/" 2>/dev/null || true
gsutil mb -l "$GCS_LOCATION" "gs://$GCS_BUCKET_NAME-llvm/" 2>/dev/null || true
gsutil mb -l "$GCS_LOCATION" "gs://$GCS_BUCKET_NAME-cache/" 2>/dev/null || true
gsutil mb -l "$GCS_LOCATION" "gs://$GCS_BUCKET_NAME-output/" 2>/dev/null || true

# Make bucket public read for easier access (optional)
echo ""
echo "Note: To make bucket publicly readable, run:"
echo "  gsutil iam ch allUsers:objectViewer gs://$GCS_BUCKET_NAME/"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload LLVM: ./sync-llvm-to-cloud.sh"
echo "  2. Provision VM: ./gcp-provision.sh"
echo "  3. Build: ./gcp-build.sh"
echo ""
echo "Cost estimate:"
echo "  - VM (e2-standard-8): ~$0.38/hour"
echo "  - Storage: ~$0.02/GB/month"
echo "  - Egress: ~$0.12/GB"
