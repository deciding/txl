# Triton Xtra Language

## Install
```
pip install -r requirements.txt # torch must be installed before hand
bash tools/cp_to_triton.sh

pip install -r thirdparty/triton/python/requirements.txt
pip install -e thirdparty/triton/python/
# CXX=/usr/bin/c++ CC=/usr/bin/cc, set them properly if pip install failed


# TEST
export PYTHONPATH=$(pwd)/python/
#if get 'GLIBCXX_3.4.30' not found, do `conda install -c conda-forge gcc=12.1.0`
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump TRITON_ALWAYS_COMPILE=1 python python/txl/tests/01-vector-add.py
```


## DEBUG
```
TRITON_LLVM_DEBUG_ONLY="triton-gpu-taskid-propagate" TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump TRITON_ALWAYS_COMPILE=1 python python/txl/tests/fused-attention.py

CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_LAUNCH_BLOCKING=1 
cuda-gdb
```

## Milestones
- [x] when running ws persistent on 8192x8192x512 (default in triton) get 411 vs. 424 TFLOPS on H100 PCIe. better than 403 of fully ws triton (2%). reached 97% of cublas.

## Changelog
```
patch folder

compiler/compiler.py
thirdparty/nvidia/backend/compiler.py: stages
runtime/jit.py: JITFunction
language/core.py
language/semantic.py
```
