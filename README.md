# Triton Xtra Language

## Install
```
cd txl
# use uv or conda
# TODO: make this test only
pip install -r requirements.txt # torch must be installed before hand

# Option1: Build
bash tools/cp_to_triton.sh
pip install -r thirdparty/triton/python/requirements.txt
cd thirdparty/triton/ && python setup.py bdist_wheel
# CXX=/usr/bin/c++ CC=/usr/bin/cc, set them properly if pip install failed

# Option2: Or directly install from wheel
pip uninstall triton # must not conflict
pip install <release>.whl


# TEST
#if get 'GLIBCXX_3.4.30' not found, do install gcc 12.1.0 or if you use conda, run `conda install -c conda-forge gcc=12.1.0`
CUDA_VISIBLE_DEVICES=1 TRITON_ALWAYS_COMPILE=1 python python/txl/tutorials/02-fused-attention.py
TRITON_PRINT_AUTOTUNING=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump TRITON_ALWAYS_COMPILE=1 python python/txl/tests/01-vector-add.py
```

## Modal

```
# 1. download from https://0x0.st/KqGx.whl, rename as txl-3.4.0-cp312-cp312-linux_x86_64.whl
# 2. go to docker/
# You have $30 budget. You can use H100, H200, B200. Have fun!
modal run flash_attention.py
```

## DEV
```
cd thirdparty/triton
python setup.py bdist_wheel
curl -F"file=@dist/txl-3.4.0-cp312-cp312-linux_x86_64.whl" https://0x0.st
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
- TXL2: 482, 526, 559, 576
- TXL3: 485, 537, 566, 594, 603
- Cute:  521, 572, 609, 628, 635


WS3 downgrade might because of the additional bar.sync added.

Known that triton optimized with 3 multi-stages with MMAPV overlap with prev buffer's MMAQK, even no warpgroup specialization.

## Milestones
- [x] when running ws persistent on 8192x8192x512 (default in triton) get 411 vs. 424 TFLOPS on H100 PCIe. better than 403 of fully ws triton (2%). reached 97% of cublas.
- [x] FA3 95%
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
