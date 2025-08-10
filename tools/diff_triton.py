# must at the src_commit
FROM_DIR='.'
PATCH_DIR="patch/triton"
NEW_PATCH_DIR="patch/triton3.4.x"

REPO_PATH = '/ssd2/zhangzn/triton/'
NEW_REPO_PATH = '/ssd2/zhangzn/triton-latest/'
src_commit  = "release/3.3.x"
trg_commit  = "release/3.4.x"

files = []
files.append(f"{FROM_DIR}/bin/RegisterTritonDialects.h")

files.append(f"{FROM_DIR}/include/CMakeLists.txt")

files.append(f"{FROM_DIR}/include/triton/Analysis/TXLUtility.h")

files.append(f"{FROM_DIR}/include/triton/Conversion/TritonToTritonGPU/Passes.td")

#include/txl

files.append(f"{FROM_DIR}/lib/Analysis/Allocation.cpp")
files.append(f"{FROM_DIR}/lib/Analysis/AxisInfo.cpp")
files.append(f"{FROM_DIR}/lib/Analysis/CMakeLists.txt")
files.append(f"{FROM_DIR}/lib/Analysis/Membar.cpp")
files.append(f"{FROM_DIR}/lib/Analysis/TXLUtility.cpp")

files.append(f"{FROM_DIR}/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp")
files.append(f"{FROM_DIR}/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp")

files.append(f"{FROM_DIR}/lib/Dialect/CMakeLists.txt")

#lib/Dialect/TXL

files.append(f"{FROM_DIR}/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/TaskIdPropagate.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/Utility.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/WSCodePartition.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/WSDataPartition.cpp")
files.append(f"{FROM_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeLists.txt")
files.append(f"{FROM_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp")

files.append(f"{FROM_DIR}/python/src/ir.cc")
files.append(f"{FROM_DIR}/python/src/passes.cc")

files.append(f"{FROM_DIR}/third_party/nvidia/backend/compiler.py")

files.append(f"{FROM_DIR}/third_party/nvidia/include/CMakeLists.txt")
files.append(f"{FROM_DIR}/third_party/nvidia/include/Dialect/CMakeLists.txt")

#third_party/nvidia/include/Dialect/TXLGPU
#third_party/nvidia/include/TXLGPUToLLVM

files.append(f"{FROM_DIR}/third_party/nvidia/lib/CMakeLists.txt")
files.append(f"{FROM_DIR}/third_party/nvidia/lib/Dialect/CMakeLists.txt")

#third_party/nvidia/lib/Dialect/TXLGPU
#third_party/nvidia/lib/TXLGPUToLLVM

files.append(f"{FROM_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp")
files.append(f"{FROM_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp")

files.append(f"{FROM_DIR}/third_party/nvidia/triton_nvidia.cc")

import sys
import os
import shutil
import subprocess
from pathlib import Path

def git_diff_files(file1, file2):
    """Diff two files in a git repository"""
    try:
        result = subprocess.run(
            ['git', 'diff', '--no-index', file1, file2],
            stdout=sys.stdout, stderr=sys.stderr
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # git diff returns non-zero when there are differences, which is normal
        if e.returncode == 1:
            return e.stdout
        print(f"Error diffing files: {e.stderr}")
        return None

def git_diff_commits(repo_path, file_path, commit1, commit2):
    """Diff a file between two commits in a git repository"""
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'diff', commit1, commit2, '--', file_path],
            stdout=sys.stdout, stderr=sys.stderr
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # git diff returns non-zero when there are differences, which is normal
        if e.returncode == 1:
            return e.stdout
        print(f"Error diffing files: {e.stderr}")
        return None

init = 32
for idx, fn in enumerate(files[init:]):
    idx += init
    txl_full_path = os.path.join(os.getcwd(), PATCH_DIR, fn)
    triton_full_path = os.path.join(REPO_PATH, fn)
    new_triton_full_path = os.path.join(NEW_REPO_PATH, fn)
    print(f"{idx}. Release Diff: {fn}")
    (git_diff_commits(REPO_PATH, fn, src_commit, trg_commit))
    print(f"{idx}. TXL Diff: {fn}")
    (git_diff_files(triton_full_path, txl_full_path))

    new_triton_path = os.path.join(os.getcwd(), NEW_PATCH_DIR, fn)
    if (os.path.exists(new_triton_full_path) and not os.path.exists(new_triton_path)):
        new_parent_path = os.path.dirname(new_triton_path)
        os.makedirs(new_parent_path, exist_ok=True)
        shutil.copy(new_triton_full_path, new_parent_path)
    print(f"vim -O {new_triton_path} {txl_full_path}")

    cont = input("CONTINUE?")
    if cont.strip() != 'q':
        continue
    else:
        break

# CHECK CHANGE OF IR
files = [
        "include/triton/Dialect/Triton/IR/TritonOps.td",
        "include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td",
        "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td",
        "third_party/nvidia/include/Dialect/NVGPU/IR/NVGPUOps.td",
]
init=4
for idx, fn in enumerate(files[init:]):
    idx += init
    print(f"{idx}. Release Diff: {fn}")
    (git_diff_commits(REPO_PATH, fn, src_commit, trg_commit))

    cont = input("CONTINUE?")
    if cont.strip() != 'q':
        continue
    else:
        break

# CHECK Base passes I used from triton
#include/txl
#lib/Dialect/TXL
#third_party/nvidia/include/Dialect/TXLGPU
#third_party/nvidia/include/TXLGPUToLLVM
#third_party/nvidia/lib/Dialect/TXLGPU
files = []
files.append(
    (
        f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp",
        f"{FROM_DIR}/third_party/nvidia/lib/Dialect/TXLGPU/Transforms/AccelerateMatmul.cpp",
    )
)
files.append(
    (
        f"{FROM_DIR}/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp",
        f"{FROM_DIR}/third_party/nvidia/lib/Dialect/TXLGPU/Transforms/OptimizeDotOperands.cpp",
    )
)
# Also WGMMAPipeline for asyncDot and updateWaits
init = 1
# remember to copy the whole folder to new patch, and remove the specific files
for idx, (triton_fn, txl_fn) in enumerate(files[init:]):
    idx += init
    txl_full_path = os.path.join(os.getcwd(), PATCH_DIR, txl_fn)
    triton_full_path = os.path.join(REPO_PATH, triton_fn)
    new_triton_full_path = os.path.join(NEW_REPO_PATH, triton_fn)
    print(f"{idx}. Release Diff: {txl_fn}")
    (git_diff_commits(REPO_PATH, triton_fn, src_commit, trg_commit))
    print(f"{idx}. TXL Diff: {txl_fn}")
    (git_diff_files(triton_full_path, txl_full_path))

    new_triton_path = os.path.join(os.getcwd(), NEW_PATCH_DIR, triton_fn)
    if (os.path.exists(new_triton_full_path) and not os.path.exists(new_triton_path)):
        new_parent_path = os.path.dirname(new_triton_path)
        os.makedirs(new_parent_path, exist_ok=True)
        shutil.copy(new_triton_full_path, new_parent_path)
    print(f"vim -O {new_triton_path} {txl_full_path}")

    cont = input("CONTINUE?")
    if cont.strip() != 'q':
        continue
    else:
        break
#third_party/nvidia/lib/TXLGPUToLLVM

# CHECK runtime.jit and compiler.compiler
# CHECK semantic and core, language/__init__.py

# CHECK proton, autotune, code_generator

# align compiler.compiler with nvidia backend
