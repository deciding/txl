#!/bin/bash
# GCP Provision Script - Create VM and attach persistent storage
# Usage: ./gcp-provision.sh [--start-existing]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp-config.sh"

START_EXISTING=false
if [[ "$1" == "--start-existing" ]]; then
    START_EXISTING=true
fi

echo "=== GCP TXL Provision ==="

# Check if VM already exists
VM_EXISTS=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    --format="value(name)" 2>/dev/null || echo "")

if [[ -n "$VM_EXISTS" ]]; then
    echo "VM $VM_NAME already exists"

    if [[ "$START_EXISTING" == "true" ]]; then
        echo "Starting existing VM..."
        gcloud compute instances start "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT"
    else
        echo "VM is already running or stopped"
        echo "Use --start-existing to start it"
    fi
else
    echo "Creating new VM: $VM_NAME"
    
    # Create startup script
    cat > /tmp/startup-script.sh << 'STARTUP'
#!/bin/bash
set -e

# Install Docker
apt-get update
apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Mount persistent disk if exists
if [ -b /dev/sdb ]; then
    mkfs.ext4 -F /dev/sdb
    mkdir -p /mnt/txl-cache
    mount /dev/sdb /mnt/txl-cache
    echo "/dev/sdb /mnt/txl-cache ext4 defaults,nofail 0 2" >> /etc/fstab
fi

# Create docker group and add current user
groupadd docker || true
usermod -aG docker $USER

systemctl start docker
systemctl enable docker

echo "Startup complete"
STARTUP

    # Create the VM
    gcloud compute instances create "$VM_NAME" \
        --zone="$GCP_ZONE" \
        --machine-type="$VM_MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size="$VM_BOOT_DISK_SIZE" \
        --disk="name=${VM_NAME}-data,mode=rw,size=$VM_DISK_SIZE,device-name=data" \
        --service-account="$SERVICE_ACCOUNT" \
        --metadata-from-file startup-script=/tmp/startup-script.sh \
        --tags=http-server,https-server \
        --quiet

    rm /tmp/startup-script.sh

    echo ""
    echo "VM created. Waiting for startup to complete..."
    sleep 30
fi

# Wait for VM to be ready
echo "Waiting for VM to be ready..."
for i in {1..30}; do
    if gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" -- "echo 'VM ready'" &> /dev/null; then
        echo "VM is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 10
done

# Get external IP
VM_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$GCP_ZONE" \
    --project="$GCP_PROJECT" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "=== VM Provisioned ==="
echo "Name: $VM_NAME"
echo "IP: $VM_IP"
echo "Zone: $GCP_ZONE"
echo ""
echo "Next steps:"
echo "  1. Upload LLVM: ./sync-llvm-to-cloud.sh"
echo "  2. Build: ./gcp-build.sh"
