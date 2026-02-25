#!/bin/bash
# Script to build TXL wheel using Docker on macOS/Linux
# Outputs wheel to ./output directory
#
# Usage:
#   ./build-wheel-docker.sh          # Build with default settings
#   ./build-wheel-docker.sh --no-cache  # Rebuild without cache
#   ./build-wheel-docker.sh --help   # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Parse arguments
DOCKER_BUILD_ARGS=""
MAX_JOBS=8
TRITON_BUILD_WITH_O1=1
TRITON_BUILD_PROTON=0
TRITON_BUILD_WITH_CLANG_LLD=1
REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            DOCKER_BUILD_ARGS="--no-cache"
            shift
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --clang)
            # Note: clang is now default, this flag is kept for backward compatibility
            TRITON_BUILD_WITH_CLANG_LLD=1
            shift
            ;;
        -r|--rebuild)
            REBUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cache       Rebuild Docker image without cache"
            echo "  -j, --jobs N     Number of parallel build jobs (default: 8)"
            echo "  -r, --rebuild   Rebuild using existing container (fast, incremental)"
            echo "  --clang          Use clang + lld instead of gcc (less memory)"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Output:"
            echo "  Wheel file will be placed in: $OUTPUT_DIR"
            echo ""
            echo "Examples:"
            echo "  $0                    # Full build with 8 jobs, O1, no proton"
            echo "  $0 -j 4               # Build with 4 jobs (less memory)"
            echo "  $0 --clang            # Use clang (less memory)"
            echo "  $0 -r                 # Fast rebuild using existing container"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== TXL Docker Wheel Builder ==="
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    echo "Please start Docker Desktop or Docker daemon"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if TXL source exists
if [ ! -f "$PROJECT_ROOT/tools/cp_to_triton.sh" ]; then
    echo "Error: TXL source not found at $PROJECT_ROOT"
    exit 1
fi

# Check if pre-downloaded LLVM exists
LLVM_DIR="$PROJECT_ROOT/llvm-7d5de303-ubuntu-x64"
if [ -d "$LLVM_DIR" ]; then
    echo "Using pre-downloaded LLVM from: $LLVM_DIR"
fi

# Check if existing container should be used for rebuild
if [ "$REBUILD" == "true" ]; then
    echo ""
    echo "=== Rebuild Mode ==="
    
    # Check if container exists
    if ! docker ps -a --format '{{.Names}}' | grep -q "^txl-wheel-build$"; then
        echo "Error: Container txl-wheel-build not found"
        echo "Run without -r flag to create a new build"
        exit 1
    fi
    
    # Start the container
    echo "Starting existing container..."
    docker start txl-wheel-build
    
    # Run incremental build
    echo "Running incremental build..."
    docker exec -e MAX_JOBS=$MAX_JOBS -e TRITON_BUILD_WITH_O1=$TRITON_BUILD_WITH_O1 \
        txl-wheel-build bash -c '
        set -e
        source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
            source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
        conda activate txl
        
        export CC=x86_64-conda-linux-gnu-gcc
        export CXX=x86_64-conda-linux-gnu-g++
        export LD_LIBRARY_PATH=/opt/conda/envs/txl/lib:/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
        
        cd /txl/thirdparty/triton
        
        # Check if build directory exists
        if [ ! -d "build" ]; then
            echo "Error: build directory not found. Run full build first."
            exit 1
        fi
        
        # Run incremental build using ninja (only rebuilds changed files)
        echo "Running incremental build with MAX_JOBS=$MAX_JOBS..."
        cd build/cmake.linux-x86_64-cpython-312
        ninja -j $MAX_JOBS
        
        # Create wheel
        cd /txl/thirdparty/triton
        python setup.py bdist_wheel --skip-build
        
        # Copy to output
        cp dist/*.whl /output/
        
        echo "Rebuild complete!"
        '
    
    echo ""
    echo "=== Rebuild complete ==="
    echo "Wheel(s) available in: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"/*.whl
    exit 0
fi

# Full build (default)
echo "=== Building Docker image ==="
echo "This may take a few minutes on first build..."
docker build $DOCKER_BUILD_ARGS \
    -f "$SCRIPT_DIR/Dockerfile.txl-wheel" \
    -t txl-wheel-builder \
    "$PROJECT_ROOT"

echo ""
echo "=== Running build container ==="
echo "Building wheel (this may take 10-30 minutes)..."

# Run container to build wheel
# Note: GPU is not needed for building the wheel, only for runtime
# Don't use --rm so the container persists for cache
# Mount LLVM directory and set LLVM_SYSPATH to use pre-downloaded LLVM
DOCKER_RUN_CMD="docker run \
    --name txl-wheel-build \
    -v "$PROJECT_ROOT:/txl" \
    -v "$OUTPUT_DIR:/output""

# Add LLVM mount if pre-downloaded LLVM exists
if [ -d "$LLVM_DIR" ]; then
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
    -v "$LLVM_DIR:/llvm:ro""
fi

# Add environment variables for LLVM (if LLVM exists)
if [ -d "$LLVM_DIR" ]; then
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
    -e LLVM_SYSPATH=/llvm"
fi

# Add MAX_JOBS for parallel compilation
DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
-e MAX_JOBS=$MAX_JOBS"

# Add memory optimization flags
DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
-e TRITON_BUILD_WITH_O1=$TRITON_BUILD_WITH_O1 \
-e TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON \
-e TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD"

echo "Building with MAX_JOBS=$MAX_JOBS, TRITON_BUILD_WITH_O1=$TRITON_BUILD_WITH_O1, TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON, TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD (clang enabled by default)"

# Run the container
eval $DOCKER_RUN_CMD txl-wheel-builder

echo ""
echo "=== Build complete ==="
echo "Wheel(s) available in: $OUTPUT_DIR"
if ls "$OUTPUT_DIR"/*.whl 1> /dev/null 2>&1; then
    ls -lh "$OUTPUT_DIR"/*.whl
else
    echo "Warning: No wheel files found in output directory"
    echo "Check container logs for errors"
    exit 1
fi
