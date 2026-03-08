#!/bin/bash
# Build script for TXL wheel inside Docker container
# This script runs inside the Docker container to build the TXL wheel

set -e  # Exit on any error

# Parse arguments
CLEAN_BUILD=false
NEW_COPY=false
while getopts "cn" opt; do
    case $opt in
        c) CLEAN_BUILD=true ;;
        n) NEW_COPY=true ;;
        \?) echo "Usage: $0 [-c] [-n]" >&2
            echo "  -c  Clean build directories before build" >&2
            echo "  -n  Run cp_to_triton.sh (apply TXL patches)" >&2
            exit 1 ;;
    esac
done

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

# Step 3: Apply TXL patches to Triton (only with -n flag)
if [ "$NEW_COPY" == "true" ]; then
    echo "=== Step 3: Applying TXL patches to Triton ==="
    bash tools/cp_to_triton.sh
else
    echo "=== Step 3: Skipping TXL patches (use -n to apply) ==="
fi

# Step 4: Install Triton build dependencies
echo "=== Step 4: Installing Triton build dependencies ==="
pip install -r thirdparty/triton/python/requirements.txt

# Step 5: Rename triton to teraxlang_triton and update imports
echo "=== Step 5: Renaming triton to teraxlang_triton ==="
cd thirdparty/triton/python

# Copy triton to teraxlang_triton
echo "Copying triton to teraxlang_triton..."
rm -rf teraxlang_triton
cp -r triton teraxlang_triton

# Replace imports in teraxlang_triton
echo "Replacing imports in teraxlang_triton..."
python /txl/tools/replace_imports.py teraxlang_triton

# Replace imports in txl (but exclude tests)
echo "Replacing imports in txl..."
rm -rf txl_temp
cp -r txl txl_temp
rm -rf txl
mv txl_temp txl
python /txl/tools/replace_imports.py txl

# Update setup.py to build teraxlang_triton instead of triton
echo "Updating setup.py..."
cat > setup.py << 'SETUP_EOF'
import os
from setuptools import setup, find_packages

# Read version from triton/__init__.py
version = {}
with open("teraxlang_triton/__init__.py") as f:
    exec(f.read(), version)

# Read requirements
install_requires = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="teraxlang",
    version=version.get("__version__", "3.5.1"),
    description="TeraXLang - CUDA kernel DSL built on Triton",
    author="TeraXLang Team",
    packages=find_packages(include=["teraxlang_triton", "teraxlang_triton.*", "txl", "txl.*"]),
    package_dir={
        "teraxlang_triton": "teraxlang_triton",
        "txl": "txl",
    },
    install_requires=install_requires,
    python_requires=">=3.8",
)
SETUP_EOF

# Go back to triton directory
cd ..

# Step 6: Build the wheel
echo "=== Step 6: Building wheel ==="
cd thirdparty/triton

# Clean build directory if -c flag is set
if [ "$CLEAN_BUILD" == "true" ]; then
    echo "Cleaning build directories..."
    rm -rf build dist
    mkdir -p build dist
fi

# Build the wheel with explicit compiler settings
echo "Building with CC=$CC CXX=$CXX"
python setup.py bdist_wheel

# Step 7: Verify and repair wheel for manylinux compliance
echo "=== Step 7: Verifying and repairing wheel ==="
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
        # Rename wheel from teraxlang to txl for backward compatibility
        BASENAME=$(basename "$FINAL_WHEEL")
        NEW_NAME=${BASENAME/teraxlang/txl}
        cp "$FINAL_WHEEL" "/output/$NEW_NAME"
        echo "Wheel copied to /output/$NEW_NAME"
        
        # Also copy with teraxlang name
        cp "$FINAL_WHEEL" "/output/$BASENAME"
        echo "Also copied as /output/$BASENAME"
        ls -lh /output/
    fi
else
    echo "Error: No wheel file found in dist/"
    ls -la dist/ 2>/dev/null || true
    exit 1
fi

echo "=== Build completed successfully ==="
echo "Date: $(date)"
