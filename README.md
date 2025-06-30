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
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump TRITON_ALWAYS_COMPILE=1 python python/txl/tests/01-vector-add.py
```


## DEBUG
```
TRITON_LLVM_DEBUG_ONLY="triton-gpu-taskid-propagate" TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump TRITON_ALWAYS_COMPILE=1 python python/txl/tests/fused-attention.py

CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_LAUNCH_BLOCKING=1 
cuda-gdb
```

## Changelog
```
patch folder

compiler/compiler.py
thirdparty/nvidia/backend/compiler.py: stages
runtime/jit.py: JITFunction
language/core.py
language/semantic.py
```
