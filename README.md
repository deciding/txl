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

## Flash-Attention-3

seq len 1k 2k 4k 8k, (batch\_size = 16k/seqlen)
num heads 32, head dim 128, causal=False
hardware: H100 PCIe

TFLOPS:
- Triton: 251, 265, 275, 279
- FA3: 372,397, 412, 423
- MS-ours: 272, 291, 299, 304
- WS1-pure: 313, 347, 367, 377
- WS2-pingpong: 301, 339, 361, 370




Known that triton optimized with 3 multi-stages with MMAPV overlap with prev buffer's MMAQK, even no warpgroup specialization.

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
