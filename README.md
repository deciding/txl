# Triton Xtra Language

## Install
```
pip install -r requirements.txt # torch must be installed before hand
bash tools/cp_to_triton.sh

pip install -r thirdparty/triton/python/requirements.txt

# Option1: Build
pip install -e thirdparty/triton/
export PYTHONPATH=$(pwd)/python/
# CXX=/usr/bin/c++ CC=/usr/bin/cc, set them properly if pip install failed
# Option2: Or directly install from wheel
pip uninstall triton # must not conflict
pip install <release>.whl


# TEST
#if get 'GLIBCXX_3.4.30' not found, do `conda install -c conda-forge gcc=12.1.0`
CUDA_VISIBLE_DEVICES=1 TRITON_ALWAYS_COMPILE=1 python python/txl/tutorials/02-fused-attention.py
TRITON_PRINT_AUTOTUNING=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump TRITON_ALWAYS_COMPILE=1 python python/txl/tests/01-vector-add.py
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
- FA3: 372,397, 412, 423
- FA3py: 317 367 388 406
- Triton: 259, 288, 287, 294
- MS-ours: 281, 297, 308, 313
- WS1-ours: 298, 333, 351, 359
- WS2-ours: 309, 347, 367, 378
- WS3-ours: 267, 296, 310, 324

WS3 downgrade might because of the additional bar.sync added.

Known that triton optimized with 3 multi-stages with MMAPV overlap with prev buffer's MMAQK, even no warpgroup specialization.

## Milestones
- [x] when running ws persistent on 8192x8192x512 (default in triton) get 411 vs. 424 TFLOPS on H100 PCIe. better than 403 of fully ws triton (2%). reached 97% of cublas.
- [x] FA3 90%
- [x] 421 vs. 340 vs. 318 (cublas vs. triton vs. txl) for multi-stage MM without TMA. need to find the reason of downgrade
- [x] Triton upgrade to 3.4

## Changelog
```
patch folder

compiler/compiler.py
thirdparty/nvidia/backend/compiler.py: stages
runtime/jit.py: JITFunction
language/core.py
language/semantic.py
```
