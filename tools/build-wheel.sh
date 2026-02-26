#!/bin/bash
# Build script for TXL wheel inside Docker container
# This script runs inside the Docker container to build the TXL wheel

set -e  # Exit on any error

# Initialize conda
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate txl

# Set compiler paths (use conda compiler names, not full paths)
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export PATH=/usr/bin:/opt/conda/bin:/opt/conda/envs/txl/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/envs/txl/lib:/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# Set MAX_JOBS for parallel compilation (default 8, override with MAX_JOBS env var)
export MAX_JOBS=${MAX_JOBS:-8}

# Memory optimization flags
# Use O1 optimization instead of default (saves memory during build)
export TRITON_BUILD_WITH_O1=${TRITON_BUILD_WITH_O1:-1}

# Use clang + lld (uses less memory than gcc, optional - set to 1 to enable)
export TRITON_BUILD_WITH_CLANG_LLD=${TRITON_BUILD_WITH_CLANG_LLD:-0}

# Disable proton to save build time and memory (optional)
export TRITON_BUILD_PROTON=${TRITON_BUILD_PROTON:-1}

# Disable ccache to avoid warning
export TRITON_BUILD_WITH_CCACHE=OFF

echo "=== TXL Wheel Build Script ==="
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "GCC: $(gcc --version | head -1)"
echo "CXX: $(g++ --version | head -1)"

# Check if TXL source is mounted at /txl
if [ -d "/txl" ]; then
    TXL_ROOT="/txl"
    echo "Using mounted TXL source from /txl"
elif [ -d "/build" ]; then
    # Check if we're in the build context
    if [ -f "/build/thirdparty/triton/setup.py" ]; then
        TXL_ROOT="/build"
        echo "Using TXL source from /build"
    else
        echo "Error: Cannot find TXL source directory"
        echo "Expected either /txl (mounted) or /build with source"
        ls -la /build 2>/dev/null || true
        exit 1
    fi
else
    echo "Error: TXL source directory not found"
    exit 1
fi

echo "TXL root: $TXL_ROOT"
cd "$TXL_ROOT"

# Verify critical files exist
if [ ! -f "tools/cp_to_triton.sh" ]; then
    echo "Error: tools/cp_to_triton.sh not found"
    exit 1
fi

if [ ! -f "thirdparty/triton/setup.py" ]; then
    echo "Error: thirdparty/triton/setup.py not found"
    echo "Make sure git submodules are initialized"
    exit 1
fi

# Step 1: Update submodules
echo "=== Step 1: Updating git submodules ==="
git submodule update --init --recursive || true

# Step 2: Install Python dependencies
echo "=== Step 2: Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || {
    echo "Warning: Some dependencies may have failed (e.g., nvidia-cutlass-dsl)"
    echo "Continuing..."
}

# Step 3: Apply TXL patches to Triton
echo "=== Step 3: Applying TXL patches to Triton ==="
bash tools/cp_to_triton.sh

# Step 4: Install Triton build dependencies
echo "=== Step 4: Installing Triton build dependencies ==="
pip install -r thirdparty/triton/python/requirements.txt

# Step 5: Build the wheel
echo "=== Step 5: Building wheel ==="
cd thirdparty/triton

# Ensure build directory exists and is clean
rm -rf build dist
mkdir -p build dist

# Build the wheel with explicit compiler settings
echo "Building with CC=$CC CXX=$CXX"
python setup.py bdist_wheel

# Step 6: Verify and repair wheel for manylinux compliance
echo "=== Step 6: Verifying and repairing wheel ==="
WHEEL_PATH=$(find dist -name "*.whl" | head -1)
if [ -n "$WHEEL_PATH" ]; then
    echo "Wheel built: $WHEEL_PATH"
    ls -lh "$WHEEL_PATH"
    
    # Run auditwheel for manylinux compliance
    echo "Running auditwheel repair..."
    auditwheel repair "$WHEEL_PATH" -w dist || {
        echo "Warning: auditwheel repair failed, using original wheel"
    }
    
    # Find final wheel
    FINAL_WHEEL=$(find dist -name "*.whl" | grep -v "none-any" | head -1)
    if [ -z "$FINAL_WHEEL" ]; then
        FINAL_WHEEL=$(find dist -name "*.whl" | head -1)
    fi
    
    echo "Final wheel: $FINAL_WHEEL"
    ls -lh "$FINAL_WHEEL"
    
    # Copy to output directory
    if [ -d "/output" ]; then
        cp "$FINAL_WHEEL" /output/
        echo "Wheel copied to /output/"
        ls -lh /output/
    fi
else
    echo "Error: No wheel file found in dist/"
    ls -la dist/ 2>/dev/null || true
    exit 1
fi

echo "=== Build completed successfully ==="
echo "Date: $(date)"
