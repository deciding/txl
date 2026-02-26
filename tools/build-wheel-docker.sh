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
TRITON_BUILD_PROTON=1
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
echo "Conda home: $CONDA_HOME_DIR (persists across container rebuilds)"
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

# Conda home directory (persist across container rebuilds)
CONDA_HOME_DIR="$PROJECT_ROOT/txl-conda"

# Create conda home directory if it doesn't exist
if [ ! -d "$CONDA_HOME_DIR" ]; then
    echo "Creating conda home directory at: $CONDA_HOME_DIR"
    mkdir -p "$CONDA_HOME_DIR"
fi

# Check if existing container should be used for rebuild
if [ "$REBUILD" == "true" ]; then
    echo ""
    echo "=== Rebuild Mode ==="
    
    # Check if persisted conda exists
    if [ ! -d "$CONDA_HOME_DIR/envs/txl" ]; then
        echo "Error: No persisted conda found in $CONDA_HOME_DIR"
        echo "Run a full build first (without -r) to create the conda environment"
        exit 1
    fi
    
    echo "Found persisted conda in $CONDA_HOME_DIR"
    echo "Starting new container with persisted conda..."
    
    # Build docker run command with persisted conda
    DOCKER_RUN_CMD="docker run \
        --name txl-wheel-build \
        -v "$PROJECT_ROOT:/txl" \
        -v "$OUTPUT_DIR:/output" \
        -v "$CONDA_HOME_DIR:/opt/conda" \
        -e CONDA_PREFIX=/opt/conda/envs/txl"
    
    # Add LLVM mount if exists
    if [ -d "$LLVM_DIR" ]; then
        DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
        -v "$LLVM_DIR:/llvm:ro""
    fi
    
    # Add LLVM environment variable
    if [ -d "$LLVM_DIR" ]; then
        DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
        -e LLVM_SYSPATH=/llvm"
    fi
    
    # Add build flags
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
    -e MAX_JOBS=$MAX_JOBS \
    -e TRITON_BUILD_WITH_O1=$TRITON_BUILD_WITH_O1 \
    -e TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON \
    -e TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD"
    
    echo "Running build with persisted conda..."
    eval $DOCKER_RUN_CMD txl-wheel-builder
    
    # Remove container after build to free resources
    echo "Removing container to free resources..."
    docker rm txl-wheel-build
    
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
# Mount conda home to persist across container rebuilds (only if it exists and has content)
DOCKER_RUN_CMD="docker run \
    --name txl-wheel-build \
    -v "$PROJECT_ROOT:/txl" \
    -v "$OUTPUT_DIR:/output""

# Only mount conda if it has content (after first build)
if [ -d "$CONDA_HOME_DIR/envs/txl" ]; then
    echo "Mounting persisted conda from $CONDA_HOME_DIR"
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
    -v "$CONDA_HOME_DIR:/opt/conda" \
    -e CONDA_PREFIX=/opt/conda/envs/txl"
else
    echo "No persisted conda found, will install fresh conda in container"
fi

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

# After successful build, save conda to persisted directory for future rebuilds
if [ ! -d "$CONDA_HOME_DIR/envs/txl" ]; then
    echo ""
    echo "=== Saving conda environment for future rebuilds ==="
    docker cp txl-wheel-build:/opt/conda/. "$CONDA_HOME_DIR/"
    echo "Conda saved to $CONDA_HOME_DIR"
fi

# Remove container after build to free resources
echo "Removing container to free resources..."
docker rm txl-wheel-build

echo ""

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
