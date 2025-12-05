"""
Persistent Matmul
=====================
This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
Various matmul methods are included, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches.
The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.

Triton and cuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.

.. code-block:: bash

    # FP8
    python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128

    # FP16
    python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

Note that currently this tutorial will fail on devices with a small shared memory size, such as RTX-4090.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import itertools

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from contextlib import contextmanager

from typing import Optional
try:
    import txl
    Has_TXL = True
    from triton.tools.tensor_descriptor import TensorDescriptor
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    # profile
    import triton.profiler.language as pl
    import triton.profiler as proton
    from txl.language.semantic import TXLSemantic

    pl.enable_semantic("triton")
    pl.enable_semantic_obj(TXLSemantic)


    print("TXL")
except:
    class txl:
        class Config:
            def __init__ (
                self,
                config,
                num_stages=1,
                num_warps=1,
                num_warpgroups=1,
                pre_hook=None,
                ):
                pass

        @staticmethod
        def jit(use_txl=False, diff_mode='ttir', diff_select=-1, log_dir='', src_file='', launch_metadata=None):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def autotune(configs=[], key='', use_cuda_graph=False):
            def decorator(func):
                return func
            return decorator
    Has_TXL = False
    DEVICE = torch.device('cuda:0')
    print("No txl")

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w, pre_hook=pre_hook) \
        for BM in [128] \
        for BN in [256] \
        for BK in [64] \
        for s in ([3]) \
        for w in [4] \
    ]



def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    NUM_CONSUMER_GROUPS = nargs.get("NUM_CONSUMER_GROUPS", 1)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_M //= NUM_CONSUMER_GROUPS
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


def matmul_get_naive_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w, pre_hook=pre_hook) \
        for BM in [128] \
        for BN in [256] \
        for BK in [64] \
        for s in ([1]) \
        for w in [4] \
    ]
@triton.autotune(
    configs=matmul_get_naive_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
#@txl.jit(launch_metadata=_matmul_launch_metadata)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
def matmul_kernel_tma(a_desc, b_desc, c_desc,  #
                      M, N, K,  #
                      BLOCK_SIZE_M: tl.constexpr,  #
                      BLOCK_SIZE_N: tl.constexpr,  #
                      BLOCK_SIZE_K: tl.constexpr,  #
                      GROUP_SIZE_M: tl.constexpr,  #
                      FP8_OUTPUT: tl.constexpr,  #
                      WARP_SPECIALIZE: tl.constexpr,  #
                      ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


def matmul_tma(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    with proton.scope("matmul_tma"):
        matmul_kernel_tma[grid](
            a_desc, b_desc, c_desc,  #
            M, N, K,  #
            FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
            WARP_SPECIALIZE=warp_specialize,  #
        )
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                SUBTILE
            }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [128]  #
        for BN in [256]  #
        for BK in [64,]  #
        for s in ([4])  #
        for w in [8]  #
        for SUBTILE in [True]  #
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 WARP_SPECIALIZE: tl.constexpr,  #
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


def matmul_tma_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c

def matmul_descriptor_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                SUBTILE
            }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [128]  #
        for BN in [256]  #
        for BK in [64]  #
        for s in ([3])  #
        for w in [4]  #
        for SUBTILE in [False]  #
    ]

def prune_invalid_configs(configs, named_args, **kwargs):
    FLATTEN = kwargs["FLATTEN"]
    # Filter out configs where EPILOGUE_SUBTILE is true and HOPPER is true
    return [conf for conf in configs if not (conf.kwargs.get("EPILOGUE_SUBTILE", True) and FLATTEN is False)]


@triton.autotune(configs=matmul_descriptor_persistent_get_configs(), key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    FLATTEN: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def matmul_descriptor_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Hopper warpspec doesn't work with flatten
    flatten = False if (warp_specialize and is_hopper()) else True
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        FLATTEN=flatten,
    )
    return c

##########################################
# Test 00: Async Load
##########################################
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 3,
        }, num_stages=3, num_warps=4)
    ],
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  NUM_STAGES: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

@txl.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 3,
        }, num_stages=3, num_warps=4)
    ],
    key=["M", "N", "K"],
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir', log_dir='dump')
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_async_load_txl_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  NUM_STAGES: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    dtype = tl.float16
    bA = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    bB = txl.smem_alloc([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype, num_stages=NUM_STAGES)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    bufIdxW = 0

    cur_bA = txl.get_buffer(bA, bufIdxW)
    cur_bB = txl.get_buffer(bB, bufIdxW)
    txl.async_load(cur_bA, a_ptrs, mask=offs_k[None, :] < K, other=0.0)
    #x = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
    txl.async_load(cur_bB, b_ptrs, mask=offs_k[:, None] < K, other=0.0)
    bufIdxW = (bufIdxW + 1) % NUM_STAGES
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

    cur_bA = txl.get_buffer(bA, bufIdxW)
    cur_bB = txl.get_buffer(bB, bufIdxW)
    txl.async_load(cur_bA, a_ptrs, mask=offs_k[None, :] < K- BLOCK_SIZE_K, other=0.0)
    txl.async_load(cur_bB, b_ptrs, mask=offs_k[:, None] < K- BLOCK_SIZE_K, other=0.0)
    bufIdxW = (bufIdxW + 1) % NUM_STAGES
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

    bufIdxR = 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        txl.async_load_wait(2)
        cur_bA = txl.get_buffer(bA, bufIdxR)
        cur_bB = txl.get_buffer(bB, bufIdxR)

        accumulator = tl.dot(cur_bA, cur_bB, accumulator)
        txl.dot_wait(1)

        cur_bA = txl.get_buffer(bA, bufIdxW)
        cur_bB = txl.get_buffer(bB, bufIdxW)
        txl.async_load(cur_bA, a_ptrs, mask=offs_k[None, :] < K - (k+2) * BLOCK_SIZE_K, other=0.0)
        txl.async_load(cur_bB, b_ptrs, mask=offs_k[:, None] < K - (k+2) * BLOCK_SIZE_K, other=0.0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bufIdxW = (bufIdxW + 1) % NUM_STAGES

        bufIdxR = (bufIdxR + 1) % NUM_STAGES

    txl.dot_wait(0)
    #txl.async_load_wait(2)

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)

def matmul_async_load_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_async_load_txl_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

##########################################
# Test 0: Naive
##########################################

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 1,
            },
            num_stages=1,
            num_warps=4,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_naive_tma_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    phase = 0
    bufIdxP = 0
    bufIdxC = 0

    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)

        for k in range(0, num_k):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)

            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)

            txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
            txl.mbar_wait(cur_mbar, phase)

            accumulator = tl.dot(a, b.T, accumulator)
            txl.dot_wait(0)

            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            offs_k += BLOCK_SIZE_K

        # Epilogue
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)

def matmul_naive_tma_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_naive_tma_txl_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# Test 1: No WS + TMA
##########################################

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                #"BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            #num_warps=4,
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_nows_tma_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    phase = 0
    bufIdxP = 0
    bufIdxC = 0

    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        # Prologue
        for i in tl.static_range(2):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)

            bufIdxP = (bufIdxP + 1) % NUM_STAGES
            offs_k += BLOCK_SIZE_K

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)

        for k in range(0, num_k):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)

            txl.mbar_wait(cur_mbar, phase)
            accumulator = tl.dot(a, b.T, accumulator)
            txl.dot_wait(2)

            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1

            if k < num_k - 2: # TODO: pred?
                cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
                a = txl.get_buffer(a0, bufIdxP)
                b = txl.get_buffer(b0, bufIdxP)
                txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
                txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)

                bufIdxP = (bufIdxP + 1) % NUM_STAGES
                offs_k += BLOCK_SIZE_K

        # Epilogue
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)

def matmul_tma_persistent_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_nows_tma_txl_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# Test 2: WS + TMA
##########################################

filename = 'dump/3H2OJ5D5CYZY5Q4QRKILSI7DZ5IUKDPDDLUQYE6POPBUMHJUGKNA/matmul_persistent_ws_tma_txl_kernel.ptx'
@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='llir', log_dir='dump')
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='llir')
#@txl.jit(launch_metadata=_matmul_launch_metadata, src_file=filename)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_ws_tma_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    a1 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer_a0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_a1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_b0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_consumer1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mbar_consumer2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)


    if txl.is_warpgroup([0]):

        phase = 1
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)

                mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                a0_buf = txl.get_buffer(a0, bufIdx)
                a1_buf = txl.get_buffer(a1, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_c1, phase)
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)

                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)


                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase^1

    if txl.is_warpgroup([1, 2]): # TODO: else
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_p_b0, phase)

                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    accumulator = tl.dot(a0_buf, b0_buf.T, accumulator) # accumulator is reg, no contention among buffers
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]): # TODO: else test
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0: # TODO: pipelinestate
                    phase = phase^1

            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M//2, offs_bn], c)

def matmul_tma_ws_persistent_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_ws_tma_txl_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# Test 3: WS + TMA + NN
##########################################

def matmul_tma_set_block_size_hook_nn(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    NUM_CONSUMER_GROUPS = nargs.get("NUM_CONSUMER_GROUPS", 1)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_M //= NUM_CONSUMER_GROUPS
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=matmul_tma_set_block_size_hook_nn
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
#@txl.jit(launch_metadata=_matmul_launch_metadata, src_file=filename)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_ws_tma_nn_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    a1 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    #b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer_a0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_a1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_b0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_consumer1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mbar_consumer2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)


    if txl.is_warpgroup([0]):

        phase = 1
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)

                mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                a0_buf = txl.get_buffer(a0, bufIdx)
                a1_buf = txl.get_buffer(a1, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_c1, phase)
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)

                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                #txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)
                txl.tma_load(b0_buf, b_desc, [offs_k, offs_bn], mbar_p_b0)


                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase^1

    if txl.is_warpgroup([1, 2]): # TODO: else
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_p_b0, phase)

                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    #accumulator = tl.dot(a0_buf, b0_buf.T, accumulator) # accumulator is reg, no contention among buffers
                    accumulator = tl.dot(a0_buf, b0_buf, accumulator) # accumulator is reg, no contention among buffers
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]): # TODO: else test
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    #accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    accumulator = tl.dot(a1_buf, b0_buf, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0: # TODO: pipelinestate
                    phase = phase^1

            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M//2, offs_bn], c)

def matmul_tma_ws_nn_persistent_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_ws_tma_nn_txl_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# Test 4: WS + TMA2
##########################################

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_ws_tma_txl_kernel2(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    a1 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer_a0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_a1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_b0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_consumer1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mbar_consumer2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)


    if txl.is_warpgroup([0]):

        phase = 1
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)

                mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                a0_buf = txl.get_buffer(a0, bufIdx)
                a1_buf = txl.get_buffer(a1, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_c1, phase)
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)

                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)


                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase^1

    if txl.is_warpgroup([1, 2]): # TODO: else
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_p_b0, phase)

                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    accumulator = tl.dot(a0_buf, b0_buf.T, accumulator) # accumulator is reg, no contention among buffers
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]): # TODO: else test
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0: # TODO: pipelinestate
                    phase = phase^1

            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M//2, offs_bn], c)

def matmul_tma_ws_persistent_txl2(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_ws_tma_txl_kernel2[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# Test x: split mul
##########################################

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 1,
            },
            num_stages=1,
            num_warps=4,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_separate_tma_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    phase = 0
    bufIdxP = 0
    bufIdxC = 0

    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)

        for k in range(0, num_k):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)

            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)

            txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
            txl.mbar_wait(cur_mbar, phase)

            for kk in tl.static_range(0, BLOCK_SIZE_K, 32):
                aa = txl.smem_slice(a, start=kk, length=32, dim=1)
                bb = txl.smem_slice(b, start=kk, length=32, dim=1)

                accumulator = tl.dot(aa, bb.T, accumulator)
                txl.dot_wait(0)
            #aa = txl.smem_slice(a, start=0, length=64, dim=1)
            #bb = txl.smem_slice(b, start=0, length=64, dim=1)
            #accumulator = tl.dot(aa, bb.T, accumulator)
            #accumulator = tl.dot(a, b.T, accumulator)
            txl.dot_wait(0)

            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            offs_k += BLOCK_SIZE_K

        # Epilogue
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)

def matmul_separate_tma_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_separate_tma_txl_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# Test : New WS
##########################################

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
#@txl.jit(src_file="dump/1121newws/RKIIRMDFZHXRF6NCI3VDQRPJKQCTN65RG3B5CDTFKTXU34IV3NZQ/matmul_persistent_ws_tma_txl_kernel_newws.ptx", launch_metadata=_matmul_launch_metadata)
#@txl.jit(diff_mode='ttir', log_dir='dump/newws1121', launch_metadata=_matmul_launch_metadata)
#@txl.jit(diff_mode='ttgir', launch_metadata=_matmul_launch_metadata)
def matmul_persistent_ws_tma_txl_kernel_newws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    a1 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    c0 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_N], dtype=dtype)
    c1 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_N], dtype=dtype)

    mbar_producer_a0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_a1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_b0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_consumer1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mbar_consumer2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    tid = txl.tid(0)
    bid = txl.bid(0)


    if txl.is_warpgroup([0]):
        txl.reg_dealloc(40)

        phase = 1
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)

                mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                a0_buf = txl.get_buffer(a0, bufIdx)
                a1_buf = txl.get_buffer(a1, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_c1, phase)
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)

                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)


                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase^1

    if txl.is_warpgroup([1, 2]): # TODO: else
        txl.reg_alloc(232)
        phase = 0
        bufIdx = 0

        if txl.is_warpgroup([1]):
            mbar_producer_a = mbar_producer_a0
            mbar_consumer = mbar_consumer1
            ax = a0
            cx = c0
        else:
            mbar_producer_a = mbar_producer_a1
            mbar_consumer = mbar_consumer2
            ax = a1
            cx = c1

        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_p_b0, phase)

                #if txl.is_warpgroup([1]):
                #    mbar_p_a = txl.get_buffer(mbar_producer_a0, bufIdx)
                #    mbar_c = txl.get_buffer(mbar_consumer1, bufIdx)
                #    a_buf = txl.get_buffer(a0, bufIdx)
                #else:
                #    mbar_p_a = txl.get_buffer(mbar_producer_a1, bufIdx)
                #    mbar_c = txl.get_buffer(mbar_consumer2, bufIdx)
                #    a_buf = txl.get_buffer(a1, bufIdx)
                mbar_p_a = txl.get_buffer(mbar_producer_a, bufIdx)
                mbar_c = txl.get_buffer(mbar_consumer, bufIdx)
                a_buf = txl.get_buffer(ax, bufIdx)

                txl.mbar_wait(mbar_p_a, phase)
                accumulator = tl.dot(a_buf, b0_buf.T, accumulator)
                txl.dot_wait(0)
                txl.mbar_arrive(mbar_c)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0: # TODO: pipelinestate
                    phase = phase^1

            c = accumulator.to(dtype)
            cx_smem = txl.get_buffer(cx, 0)
            txl.smem_store(cx_smem, c)
            if txl.is_warpgroup([1]):
                offs_om = offs_am
                #c_desc.store([offs_om, offs_bn], c)
                txl.tma_store(cx_smem, c_desc, [offs_om, offs_bn])
            else:
                offs_om = offs_am + BLOCK_SIZE_M//2
                #c_desc.store([offs_om, offs_bn], c)
                txl.tma_store(cx_smem, c_desc, [offs_om, offs_bn])

def matmul_tma_ws_persistent_txl_newws(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_ws_tma_txl_kernel_newws[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# END OF TXL
##########################################

def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                SUBTILE
            }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [128]  #
        #for BN in [128, 256]  #
        for BN in [256]  #
        #for BK in [64, 128]  #
        for BK in [64]  #
        #for s in ([2, 3, 4])  #
        for s in ([3])  #
        #for w in [4, 8]  #
        for w in [8]  #
        #for SUBTILE in [True, False]  #
        for SUBTILE in [False]  #
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 WARP_SPECIALIZE: tl.constexpr,  #
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)

def matmul_tma_persistent(a, b, warp_specialize: bool=False):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c


##########################################
# Blackwell
##########################################

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1,
                      num_warps=8),
        ]
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
#@txl.autotune(
#    configs=get_cuda_autotune_config(),
#    key=['M', 'N', 'K'],
#)
#@txl.jit
##@txl.jit(diff_mode='ttgir', log_dir='dump/1130mm')
def matmul_kernel_bw1(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    #num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    #num_pid_in_group = GROUP_SIZE_M * num_pid_n
    #group_id = pid // num_pid_in_group
    #first_pid_m = group_id * GROUP_SIZE_M
    #group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    #pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    #pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    #tl.assume(pid_m >= 0)
    #tl.assume(pid_n >= 0)
    #tl.assume(stride_am > 0)
    #tl.assume(stride_ak > 0)
    #tl.assume(stride_bn > 0)
    #tl.assume(stride_bk > 0)
    #tl.assume(stride_cm > 0)
    #tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        #a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        #b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    #if ACTIVATION == "leaky_relu":
    #    accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    #c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    #tl.store(c_ptrs, c, mask=c_mask)
    tl.store(c_ptrs, c)

def matmul_bw1(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_bw1[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


############
# Blackwell + TXL
############

@txl.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@txl.jit
#@txl.jit(diff_mode='ttgir', log_dir='dump/1130mm')
def matmul_kernel_bw2(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    dtype = tl.float16

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    #num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    #num_pid_in_group = GROUP_SIZE_M * num_pid_n
    #group_id = pid // num_pid_in_group
    #first_pid_m = group_id * GROUP_SIZE_M
    #group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    #pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    #pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    bA_buf = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=3)
    bB_buf = txl.smem_alloc([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype, num_stages=3)

    acc_buf = txl.tmem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=1)
    acc = txl.get_buffer(acc_buf, 0)

    zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    txl.tmem_store(acc, zeros)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    #accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    bufIdx = 0
    phase = 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        #a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        #b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        #a = tl.load(a_ptrs)
        #b = tl.load(b_ptrs)
        #acc = tl.dot(a, b, acc)

        cur_bA = txl.get_buffer(bA_buf, bufIdx)
        cur_bB = txl.get_buffer(bB_buf, bufIdx)
        txl.async_load(cur_bA, a_ptrs)
        txl.async_load(cur_bB, b_ptrs)
        txl.async_load_wait(0)
        # We accumulate along the K dimension.
        #accumulator = tl.dot(a, b, accumulator)
        acc = tl.dot(cur_bA, cur_bB, acc)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        bufIdx += 1
        if bufIdx == 3:
            bufIdx = 0
            phase ^= 1
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    #if ACTIVATION == "leaky_relu":
    #    accumulator = leaky_relu(accumulator)
    #c = accumulator.to(tl.float16)
    c = txl.tmem_load(acc)
    c = c.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    #c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    #tl.store(c_ptrs, c, mask=c_mask)
    tl.store(c_ptrs, c)

def matmul_bw2(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_bw2[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

############
# Blackwell + TXL + tma + persistent
############

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            #num_warps=4,
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
@txl.jit(launch_metadata=_matmul_launch_metadata)
#@txl.jit(launch_metadata=_matmul_launch_metadata, src_file='/workspace/matmul_persistent_nows_tma_txl_bw_kernel.ptx')
def matmul_persistent_nows_tma_txl_bw_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    acc0 = txl.tmem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=2)
    zeros = txl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    #curAcc0 = txl.get_buffer(acc0, 0)
    #txl.tmem_store(curAcc0, zeros)
    #curAcc1 = txl.get_buffer(acc0, 1)
    #txl.tmem_store(curAcc1, zeros)

    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxP = 0
    bufIdxC = 0
    phase = 0

    bufIdxAcc = 0
    bufIdxAccMbar = 0
    phaseAccMbar = 0

    tid = txl.tid(0)


    #for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
    #for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        #group_id = pid // num_pid_in_group
        #first_pid_m = group_id * GROUP_SIZE_M
        #group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        #pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        #pid_n = (pid % num_pid_in_group) // group_size_m
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        # Prologue
        for i in tl.static_range(2):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)

            bufIdxP = (bufIdxP + 1) % NUM_STAGES
            offs_k += BLOCK_SIZE_K

        cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
        a = txl.get_buffer(a0, bufIdxC)
        b = txl.get_buffer(b0, bufIdxC)
        txl.mbar_wait(cur_mbar, phase)

        curAcc = txl.get_buffer(acc0, bufIdxAcc)
        txl.tmem_store(curAcc, zeros)
        curAcc = tl.dot(a, b.T, curAcc)

        #bufIdxC += 1 # TODO: assume no change of phase
        bufIdxC = (bufIdxC + 1) % NUM_STAGES
        if bufIdxC == 0:
            phase = phase ^ 1

        cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
        a = txl.get_buffer(a0, bufIdxP)
        b = txl.get_buffer(b0, bufIdxP)
        txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
        txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
        txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
        bufIdxP = (bufIdxP + 1) % NUM_STAGES
        offs_k += BLOCK_SIZE_K

        #accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)


        for k in range(0, num_k-1):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)

            txl.mbar_wait(cur_mbar, phase)

            curAcc = tl.dot(a, b.T, curAcc)

            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            #bufIdxAccMbar = (bufIdxAccMbar + 1) % 2
            #if bufIdxAccMbar == 0:
            #    phaseAccMbar = phaseAccMbar ^ 1


            if k < num_k - 3: # TODO: pred?
                cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
                a = txl.get_buffer(a0, bufIdxP)
                b = txl.get_buffer(b0, bufIdxP)
                txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
                txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)

                bufIdxP = (bufIdxP + 1) % NUM_STAGES
                offs_k += BLOCK_SIZE_K

        # Epilogue
        c = txl.tmem_load(curAcc)
        c = c.to(tl.float16)

        c_desc.store([offs_am, offs_bn], c)

        bufIdxAcc = (bufIdxAcc + 1) % 2

def matmul_persistent_nows_tma_txl_bw3(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_nows_tma_txl_bw_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

############
# Blackwell4 + TXL + tma + persistent + dotx useD + dotx mbar
############

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            #num_warps=4,
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
@txl.jit(launch_metadata=_matmul_launch_metadata)
#@txl.jit(launch_metadata=_matmul_launch_metadata, src_file='/workspace/matmul_persistent_nows_tma_txl_bw_kernel.ptx')
def matmul_persistent_nows_tma_txl_bw_kernel4(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    acc0 = txl.tmem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=2)
    zeros = txl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    #curAcc0 = txl.get_buffer(acc0, 0)
    #txl.tmem_store(curAcc0, zeros)
    #curAcc1 = txl.get_buffer(acc0, 1)
    #txl.tmem_store(curAcc1, zeros)

    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxP = 0
    bufIdxC = 0
    phase = 0

    bufIdxAcc = 0
    bufIdxAccMbar = 0
    phaseAccMbar = 0

    tid = txl.tid(0)


    #for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
    #for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        #group_id = pid // num_pid_in_group
        #first_pid_m = group_id * GROUP_SIZE_M
        #group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        #pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        #pid_n = (pid % num_pid_in_group) // group_size_m
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        # Prologue
        for i in tl.static_range(2):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)

            bufIdxP = (bufIdxP + 1) % NUM_STAGES
            offs_k += BLOCK_SIZE_K

        cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
        a = txl.get_buffer(a0, bufIdxC)
        b = txl.get_buffer(b0, bufIdxC)
        txl.mbar_wait(cur_mbar, phase)

        curAcc = txl.get_buffer(acc0, bufIdxAcc)
        txl.tmem_store(curAcc, zeros)
        curAcc = tl.dot(a, b.T, curAcc)

        #bufIdxC += 1 # TODO: assume no change of phase
        bufIdxC = (bufIdxC + 1) % NUM_STAGES
        if bufIdxC == 0:
            phase = phase ^ 1

        cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
        a = txl.get_buffer(a0, bufIdxP)
        b = txl.get_buffer(b0, bufIdxP)
        txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
        txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
        txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
        bufIdxP = (bufIdxP + 1) % NUM_STAGES
        offs_k += BLOCK_SIZE_K

        #accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)


        for k in range(0, num_k-1):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)

            txl.mbar_wait(cur_mbar, phase)

            curAcc = tl.dot(a, b.T, curAcc)

            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            #bufIdxAccMbar = (bufIdxAccMbar + 1) % 2
            #if bufIdxAccMbar == 0:
            #    phaseAccMbar = phaseAccMbar ^ 1


            if k < num_k - 3: # TODO: pred?
                cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
                a = txl.get_buffer(a0, bufIdxP)
                b = txl.get_buffer(b0, bufIdxP)
                txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
                txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)

                bufIdxP = (bufIdxP + 1) % NUM_STAGES
                offs_k += BLOCK_SIZE_K

        # Epilogue
        c = txl.tmem_load(curAcc)
        c = c.to(tl.float16)

        c_desc.store([offs_am, offs_bn], c)

        bufIdxAcc = (bufIdxAcc + 1) % 2

def matmul_persistent_nows_tma_txl_bw4(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_persistent_nows_tma_txl_bw_kernel4[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )
    return c

##########################################
# END Of Blackwell
##########################################

def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"torch [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        c = torch.matmul(a, b.T)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(label, reps, warmup_reps, fn, *args):
    print(f"Benchmarking {label}: ...", end="")
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)
    print(f"\rBenchmarking {label}: done")


def bench(K, dtype, reps=100, warmup_reps=25, algo='0'):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    bn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = bn.T.contiguous()

    if cublas is not None:
        bench_fn("cublas", reps, warmup_reps, cublas_matmul, a, b)
    #if dtype == torch.float16:
    #    bench_fn("torch", reps, warmup_reps, torch_matmul, a, b)
    #bench_fn("naive", reps, warmup_reps, matmul, a, b.T)
    #bench_fn("persistent", reps, warmup_reps, matmul_persistent, a, b.T)
    #bench_fn("tma", reps, warmup_reps, lambda a, b: matmul_tma(a, b, False), a, b)

    #bench_fn("async_load", reps, warmup_reps, matmul, a, bn)
    #bench_fn("async_load_txl", reps, warmup_reps, matmul_async_load_txl, a, bn)
    #bench_fn("naive_tma_txl", reps, warmup_reps, matmul_naive_tma_txl, a, b) #0
    #bench_fn("tma_persistent_txl", reps, warmup_reps, matmul_tma_persistent_txl, a, b) #1
    #bench_fn("tma_ws_persistent_txl", reps, warmup_reps, matmul_tma_ws_persistent_txl, a, b) #2
    #bench_fn("tma_ws_persistent_txl_newws", reps, warmup_reps, matmul_tma_ws_persistent_txl_newws, a, b) #2
    #bench_fn("tma_ws_nn_persistent_txl", reps, warmup_reps, matmul_tma_ws_nn_persistent_txl, a, bn)

    if algo == "0":
        bench_fn("tma_persistent", reps, warmup_reps, matmul_tma_persistent, a, b) #2
    elif algo == 'b1':
        bench_fn("blackwell1", reps, warmup_reps, matmul_bw1, a, bn)
    elif algo == 'b2':
        bench_fn("blackwell2", reps, warmup_reps, matmul_bw2, a, bn)
    elif algo == 'b3':
        bench_fn("blackwell_nows_tma_persistent", reps, warmup_reps, matmul_persistent_nows_tma_txl_bw3, a, b)
    return
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]
    for ws in warp_specialize:
        ws_str = "_ws" if ws else ""
        # disable on-host warpspec on Hopper
        #if HAS_HOST_TENSOR_DESC and not (is_hopper() and ws):
        #    bench_fn(f"tma_persistent{ws_str}", reps, warmup_reps, lambda a, b: matmul_tma_persistent(a, b, ws), a, b)
        #    #bench_fn(f"tma{ws_str}", reps, warmup_reps, lambda a, b: matmul_tma(a, b, ws), a, b)
        if HAS_TENSOR_DESC and ws:
            bench_fn(f"descriptor_persistent{ws_str}", reps, warmup_reps,
                     lambda a, b: matmul_descriptor_persistent(a, b, ws), a, b)


def run_test(expect, fn, a, b, label, enabled=True, log=False):
    print(f"  {label}: ...", end="")
    if enabled:
        actual = fn(a, b)
        if log:
            print()
            print(expect)
            print(actual)
            print((expect-actual).mean(dim=0))
        passed = torch.allclose(expect, actual.to(expect.dtype), atol=1.0)
        icon = "✅" if passed else "❌"
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")


def validate(M, N, K, dtype, log=False, algo='0'):
    print(f"{M=}, {N=}, {K=}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    bn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = bn.T.contiguous()

    naive_result = cublas_matmul(a, b).to(torch.float16)
    #naive_result = matmul(a, b.T).to(torch.float16)
    #run_test(naive_result, torch_matmul, a, b, "Torch", enabled=dtype == torch.float16)
    #run_test(naive_result, cublas_matmul, a, b, "cuBLAS", enabled=cublas is not None)
    #run_test(naive_result, matmul_persistent, a, b.T, "Persistent")
    #run_test(naive_result, lambda a, b: matmul_descriptor_persistent(a, b, True), a, b, "TMA WS Persistent", log=log)

    #run_test(naive_result, lambda a, b: matmul_tma(a, b, False), a, b, "TMA Naive", log=log)

    #run_test(naive_result, lambda a, b: matmul(a, b), a, bn, "AsyncLoad Naive", log=log)
    #run_test(naive_result, lambda a, b: matmul_async_load_txl(a, b), a, bn, "AsyncLoad TXL", log=log)

    #run_test(naive_result, lambda a, b: matmul_naive_tma_txl(a, b), a, b, "TXL TMA Naive", log=log)
    #run_test(naive_result, lambda a, b: matmul_tma_persistent_txl(a, b), a, b, "TXL TMA Persistent", log=log)
    #run_test(naive_result, lambda a, b: matmul_tma_ws_persistent_txl(a, b), a, b, "TXL TMA WS Persistent", log=True)
    #run_test(naive_result, lambda a, b: matmul_tma_ws_persistent_txl_newws(a, b), a, b, "TXL TMA WS Persistent", log=True)
    #run_test(naive_result, lambda a, b: matmul_tma_ws_nn_persistent_txl(a, bn), a, bn, "TXL TMA WS NN Persistent", log=log)
    #run_test(naive_result, lambda a, b: matmul_separate_tma_txl(a, b), a, b, "TXL TMA split k", log=log)

    if algo == "0":
        run_test(naive_result, lambda a, b: matmul_tma_persistent(a, b), a, b, "TMA Original Persistent", log=True)
    elif algo == 'b1':
        run_test(naive_result, lambda a, b: matmul_bw1(a, bn), a, bn, "Blackwell simple matmul", log=True)
    elif algo == 'b2':
        run_test(naive_result, lambda a, b: matmul_bw2(a, bn), a, bn, "Blackwell txl simple matmul", log=True)
    elif algo == 'b3':
        run_test(naive_result, lambda a, b: matmul_persistent_nows_tma_txl_bw3(a, b), a, b, "Blackwell txl nows tma persistent matmul", log=True)

    return

    kernels = [
        #(matmul_tma, "TMA", HAS_HOST_TENSOR_DESC),
        #(matmul_tma_persistent, "TMA Persistent", HAS_HOST_TENSOR_DESC),
        #(matmul_descriptor_persistent, "Tensor Descriptor Persistent", HAS_TENSOR_DESC),
    ]
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]

    for (kernel, label, enabled), warp_specialize in itertools.product(kernels, warp_specialize):
        label = f"{label} (warp_specialize={warp_specialize})"
        # skip if hopper and warp_specialize and not on-device
        skipped = is_hopper() and warp_specialize and kernel != matmul_descriptor_persistent
        enabled = enabled and (not warp_specialize or HAS_TENSOR_DESC) and (not skipped)
        run_test(naive_result, lambda a, b: kernel(a, b, warp_specialize), a, b, label, enabled)
    print()


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    if precision == 'fp8':
        metric_names = ["tflop8/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)

def profile(M, N, K, dtype, log=False):
    print(f"{M=}, {N=}, {K=}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    #naive_result = cublas_matmul(a, b)
    print(matmul_tma_ws_persistent_txl(a, b))


def test_matmul(dump_dir=None, algo='0'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()
    import os;print(os.getpid())

    #dump_dir='dump/1122newws/'
    #dump_dir = None

    from triton import knobs
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-wgmma-pipeline"
    knobs.runtime.override_arch='sm100'
    knobs.autotuning.print=True
    knobs.compilation.always_compile=True

    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
    else:
        dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(0)

        #validate(32, 32, 32, dtype)
        #validate(128, 128, 512, dtype, log=True)
        #validate(8192, 8192, args.K_range[0], dtype, log=True, algo=algo)
        validate(128*32, 256*16, args.K_range[0], dtype, log=True, algo=algo)

        #profile(8192, 8192, args.K_range[0], dtype)
        #exit()

        proton.start("matmul", hook="triton")
        #proton.deactivate()
        for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
            bench(K, dtype, algo=algo)
        proton.finalize()
        show_profile(args.prec, "matmul")

if __name__ == "__main__":
    #test_matmul('dump/1204mm', 'b3')
    test_matmul(None, 'b3')
