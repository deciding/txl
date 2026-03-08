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
import triton.profiler.language as pl
import triton.profiler as proton

pl.enable_semantic("triton")
try:
    import teraxlang as txl

    Has_TXL = True
    from triton.tools.tensor_descriptor import TensorDescriptor

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    from teraxlang.language.semantic import TXLSemantic

    pl.enable_semantic_obj(TXLSemantic)
    print("TXL")
except:

    class txl:
        class Config:
            def __init__(
                self, config, num_stages=1, num_warps=1, num_warpgroups=1, pre_hook=None
            ):
                pass

        @staticmethod
        def jit(
            use_txl=False,
            diff_mode="ttir",
            diff_select=-1,
            log_dir="",
            src_file="",
            launch_metadata=None,
        ):

            def decorator(func):
                return func

            return decorator

        @staticmethod
        def autotune(configs=[], key="", use_cuda_graph=False):

            def decorator(func):
                return func

            return decorator

    Has_TXL = False
    DEVICE = torch.device("cuda:0")
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
    M, N, K, WS = (args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False))
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(
    triton.tools.tensor_descriptor, "TensorDescriptor"
)
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [128]
        for BN in [256]
        for BK in [64]
        for s in [3]
        for w in [4]
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
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [128]
        for BN in [256]
        for BK in [64]
        for s in [1]
        for w in [4]
    ]


@triton.autotune(
    configs=matmul_get_naive_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
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
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    with proton.scope("matmul_tma"):
        matmul_kernel_tma[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            FP8_OUTPUT=dtype == torch.float8_e4m3fn,
            WARP_SPECIALIZE=warp_specialize,
        )
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + tile_id % group_size_m
    pid_n = tile_id % num_pid_in_group // group_size_m
    return (pid_m, pid_n)


@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
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
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=NUM_SMS,
    )
    return c


def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [128]
        for BN in [256]
        for BK in [64]
        for s in [3]
        for w in [8]
        for SUBTILE in [False]
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            pl.enter_scope("load")
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            pl.exit_scope("load")
            pl.enter_scope("dot")
            accumulator = tl.dot(a, b.T, accumulator)
            pl.exit_scope("dot")
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N
        pl.enter_scope("epi")
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
        pl.exit_scope("epi")


def matmul_tma_persistent(a, b, warp_specialize: bool = True):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_kernel_tma_persistent[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=warp_specialize,
    )
    return c


def matmul_descriptor_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [128]
        for BN in [256]
        for BK in [64]
        for s in [3]
        for w in [4]
        for SUBTILE in [False]
    ]


def prune_invalid_configs(configs, named_args, **kwargs):
    FLATTEN = kwargs["FLATTEN"]
    return [
        conf
        for conf in configs
        if not (conf.kwargs.get("EPILOGUE_SUBTILE", True) and FLATTEN is False)
    ]


@triton.autotune(
    configs=matmul_descriptor_persistent_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
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
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    flatten = False if warp_specialize and is_hopper() else True
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_descriptor_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=warp_specialize,
        FLATTEN=flatten,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
        )
    ],
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
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
    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


@txl.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
        )
    ],
    key=["M", "N", "K"],
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_async_load_txl_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
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
    bA = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    bB = txl.smem_alloc(
        [BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype, num_stages=NUM_STAGES
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    bufIdxW = 0
    cur_bA = txl.get_buffer(bA, bufIdxW)
    cur_bB = txl.get_buffer(bB, bufIdxW)
    txl.async_load(cur_bA, a_ptrs, mask=offs_k[None, :] < K, other=0.0)
    txl.async_load(cur_bB, b_ptrs, mask=offs_k[:, None] < K, other=0.0)
    bufIdxW = (bufIdxW + 1) % NUM_STAGES
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
    cur_bA = txl.get_buffer(bA, bufIdxW)
    cur_bB = txl.get_buffer(bB, bufIdxW)
    txl.async_load(cur_bA, a_ptrs, mask=offs_k[None, :] < K - BLOCK_SIZE_K, other=0.0)
    txl.async_load(cur_bB, b_ptrs, mask=offs_k[:, None] < K - BLOCK_SIZE_K, other=0.0)
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
        txl.async_load(
            cur_bA, a_ptrs, mask=offs_k[None, :] < K - (k + 2) * BLOCK_SIZE_K, other=0.0
        )
        txl.async_load(
            cur_bB, b_ptrs, mask=offs_k[:, None] < K - (k + 2) * BLOCK_SIZE_K, other=0.0
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bufIdxW = (bufIdxW + 1) % NUM_STAGES
        bufIdxR = (bufIdxR + 1) % NUM_STAGES
    txl.dot_wait(0)
    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_async_load_txl(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_async_load_txl_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


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
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode="ttgir")
def matmul_naive_tma_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    phase = 0
    bufIdxP = 0
    bufIdxC = 0
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
        pid_n = pid % num_pid_in_group // group_size_m
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)
        for k in range(0, num_k):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)
            txl.mbar_expect(
                cur_mbar,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
            txl.mbar_wait(cur_mbar, phase)
            accumulator = tl.dot(a, b.T, accumulator)
            txl.dot_wait(0)
            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            offs_k += BLOCK_SIZE_K
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)


def matmul_naive_tma_txl(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_naive_tma_txl_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


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
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
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
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    phase = 0
    bufIdxP = 0
    bufIdxC = 0
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
        pid_n = pid % num_pid_in_group // group_size_m
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0
        for i in tl.static_range(2):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(
                cur_mbar,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
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
            if k < num_k - 2:
                cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
                a = txl.get_buffer(a0, bufIdxP)
                b = txl.get_buffer(b0, bufIdxP)
                txl.mbar_expect(
                    cur_mbar,
                    BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
                )
                txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
                txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
                bufIdxP = (bufIdxP + 1) % NUM_STAGES
                offs_k += BLOCK_SIZE_K
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)


def matmul_tma_persistent_txl(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_nows_tma_txl_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


filename = "dump/3H2OJ5D5CYZY5Q4QRKILSI7DZ5IUKDPDDLUQYE6POPBUMHJUGKNA/matmul_persistent_ws_tma_txl_kernel.ptx"


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
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
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
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    a1 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
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
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
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
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)
                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N * BLOCK_SIZE_K * 2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)
                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(
                    a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1
                )
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)
                txl.mbar_wait(mbar_p_b0, phase)
                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    accumulator = tl.dot(a0_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]):
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M // 2, offs_bn], c)


def matmul_tma_ws_persistent_txl(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_ws_tma_txl_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


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
            pre_hook=matmul_tma_set_block_size_hook_nn,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
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
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    a1 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype, num_stages=NUM_STAGES
    )
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
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
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
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)
                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N * BLOCK_SIZE_K * 2)
                txl.tma_load(b0_buf, b_desc, [offs_k, offs_bn], mbar_p_b0)
                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(
                    a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1
                )
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)
                txl.mbar_wait(mbar_p_b0, phase)
                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    accumulator = tl.dot(a0_buf, b0_buf, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]):
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    accumulator = tl.dot(a1_buf, b0_buf, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M // 2, offs_bn], c)


def matmul_tma_ws_nn_persistent_txl(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_ws_tma_nn_txl_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


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
            pre_hook=matmul_tma_set_block_size_hook,
        )
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
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    a1 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
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
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
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
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)
                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N * BLOCK_SIZE_K * 2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)
                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(
                    a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1
                )
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)
                txl.mbar_wait(mbar_p_b0, phase)
                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    accumulator = tl.dot(a0_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]):
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M // 2, offs_bn], c)


def matmul_tma_ws_persistent_txl2(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_ws_tma_txl_kernel2[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


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
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
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
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    phase = 0
    bufIdxP = 0
    bufIdxC = 0
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
        pid_n = pid % num_pid_in_group // group_size_m
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        num_k = tl.cdiv(K, BLOCK_SIZE_K)
        for k in range(0, num_k):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)
            txl.mbar_expect(
                cur_mbar,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
            txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
            txl.mbar_wait(cur_mbar, phase)
            for kk in tl.static_range(0, BLOCK_SIZE_K, 32):
                aa = txl.smem_slice(a, start=kk, length=32, dim=1)
                bb = txl.smem_slice(b, start=kk, length=32, dim=1)
                accumulator = tl.dot(aa, bb.T, accumulator)
                txl.dot_wait(0)
            txl.dot_wait(0)
            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            offs_k += BLOCK_SIZE_K
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)


def matmul_separate_tma_txl(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_separate_tma_txl_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


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
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_ws_tma_txl_kernel_newws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    a1 = txl.smem_alloc(
        [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    c0 = txl.smem_alloc([BLOCK_SIZE_M // 2, BLOCK_SIZE_N], dtype=dtype)
    c1 = txl.smem_alloc([BLOCK_SIZE_M // 2, BLOCK_SIZE_N], dtype=dtype)
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
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
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
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)
                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N * BLOCK_SIZE_K * 2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)
                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M // 2 * BLOCK_SIZE_K * 2)
                txl.tma_load(
                    a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1
                )
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
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
            pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
            pid_n = pid % num_pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)
                txl.mbar_wait(mbar_p_b0, phase)
                mbar_p_a = txl.get_buffer(mbar_producer_a, bufIdx)
                mbar_c = txl.get_buffer(mbar_consumer, bufIdx)
                a_buf = txl.get_buffer(ax, bufIdx)
                txl.mbar_wait(mbar_p_a, phase)
                accumulator = tl.dot(a_buf, b0_buf.T, accumulator)
                txl.dot_wait(0)
                txl.mbar_arrive(mbar_c)
                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase ^ 1
            c = accumulator.to(dtype)
            cx_smem = txl.get_buffer(cx, 0)
            txl.smem_store(cx_smem, c)
            if txl.is_warpgroup([1]):
                offs_om = offs_am
                txl.tma_store(cx_smem, c_desc, [offs_om, offs_bn])
            else:
                offs_om = offs_am + BLOCK_SIZE_M // 2
                txl.tma_store(cx_smem, c_desc, [offs_om, offs_bn])


def matmul_tma_ws_persistent_txl_newws(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_ws_tma_txl_kernel_newws[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=8,
        )
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel_bw1(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


def matmul_bw1(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel_bw1[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )
    return c


@txl.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
@txl.jit
def matmul_kernel_bw2(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    dtype = tl.float16
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bA_buf = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=3)
    bB_buf = txl.smem_alloc([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype, num_stages=3)
    acc_buf = txl.tmem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=1
    )
    acc = txl.get_buffer(acc_buf, 0)
    zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    txl.tmem_store(acc, zeros)
    bufIdx = 0
    phase = 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        cur_bA = txl.get_buffer(bA_buf, bufIdx)
        cur_bB = txl.get_buffer(bB_buf, bufIdx)
        txl.async_load(cur_bA, a_ptrs)
        txl.async_load(cur_bB, b_ptrs)
        txl.async_load_wait(0)
        acc = tl.dot(cur_bA, cur_bB, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bufIdx += 1
        if bufIdx == 3:
            bufIdx = 0
            phase ^= 1
    c = txl.tmem_load(acc)
    c = c.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


def matmul_bw2(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel_bw2[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )
    return c


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
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_nows_tma_txl_bw_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    acc0 = txl.tmem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=2)
    zeros = txl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxP = 0
    bufIdxC = 0
    phase = 0
    bufIdxAcc = 0
    bufIdxAccMbar = 0
    phaseAccMbar = 0
    tid = txl.tid(0)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0
        for i in tl.static_range(2):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(
                cur_mbar,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
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
        bufIdxC = (bufIdxC + 1) % NUM_STAGES
        if bufIdxC == 0:
            phase = phase ^ 1
        cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
        a = txl.get_buffer(a0, bufIdxP)
        b = txl.get_buffer(b0, bufIdxP)
        txl.mbar_expect(
            cur_mbar, BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2
        )
        txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
        txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
        bufIdxP = (bufIdxP + 1) % NUM_STAGES
        offs_k += BLOCK_SIZE_K
        num_k = tl.cdiv(K, BLOCK_SIZE_K)
        for k in range(0, num_k - 1):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)
            txl.mbar_wait(cur_mbar, phase)
            curAcc = tl.dot(a, b.T, curAcc)
            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            if k < num_k - 3:
                cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
                a = txl.get_buffer(a0, bufIdxP)
                b = txl.get_buffer(b0, bufIdxP)
                txl.mbar_expect(
                    cur_mbar,
                    BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
                )
                txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
                txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
                bufIdxP = (bufIdxP + 1) % NUM_STAGES
                offs_k += BLOCK_SIZE_K
        c = txl.tmem_load(curAcc)
        c = c.to(tl.float16)
        c_desc.store([offs_am, offs_bn], c)
        bufIdxAcc = (bufIdxAcc + 1) % 2


def matmul_persistent_nows_tma_txl_bw3(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_nows_tma_txl_bw_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 4,
                "EPILOGUE_SUBTILE": True,
            },
            num_stages=4,
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_nows_tma_txl_bw_kernel4(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    acc0 = txl.tmem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=2)
    zeros = txl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxP = 0
    bufIdxC = 0
    phase = 0
    mbar_tmem = txl.mbar_alloc(1, num_stages=2)
    bufIdxAcc = 0
    bufIdxAccMbar = 0
    phaseAccMbar = 0
    newBufIdxAccMbar = 1
    tid = txl.tid(0)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0
        for i in tl.static_range(NUM_STAGES - 1):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(
                cur_mbar,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
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
        curAccMbar = txl.get_buffer(mbar_tmem, bufIdxAccMbar)
        curAcc = txl.dotx(a, b.T, curAcc, curAccMbar)
        bufIdxC = (bufIdxC + 1) % NUM_STAGES
        if bufIdxC == 0:
            phase = phase ^ 1
        cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
        a = txl.get_buffer(a0, bufIdxP)
        b = txl.get_buffer(b0, bufIdxP)
        txl.mbar_expect(
            cur_mbar, BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2
        )
        txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
        txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
        bufIdxP = (bufIdxP + 1) % NUM_STAGES
        offs_k += BLOCK_SIZE_K
        num_k = tl.cdiv(K, BLOCK_SIZE_K)
        for k in range(0, num_k - 1):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
            newAccMbar = txl.get_buffer(mbar_tmem, newBufIdxAccMbar)
            curAccMbar = txl.get_buffer(mbar_tmem, bufIdxAccMbar)
            a = txl.get_buffer(a0, bufIdxC)
            b = txl.get_buffer(b0, bufIdxC)
            txl.mbar_wait(cur_mbar, phase)
            curAcc = txl.dotx(a, b.T, curAcc, newAccMbar)
            txl.mbar_wait(curAccMbar, phaseAccMbar)
            bufIdxC = (bufIdxC + 1) % NUM_STAGES
            if bufIdxC == 0:
                phase = phase ^ 1
            bufIdxAccMbar = (bufIdxAccMbar + 1) % 2
            if bufIdxAccMbar == 0:
                phaseAccMbar = phaseAccMbar ^ 1
            newBufIdxAccMbar = (newBufIdxAccMbar + 1) % 2
            if k < num_k - NUM_STAGES:
                cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
                a = txl.get_buffer(a0, bufIdxP)
                b = txl.get_buffer(b0, bufIdxP)
                txl.mbar_expect(
                    cur_mbar,
                    BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
                )
                txl.tma_load(a, a_desc, [offs_am, offs_k], cur_mbar)
                txl.tma_load(b, b_desc, [offs_bn, offs_k], cur_mbar)
                bufIdxP = (bufIdxP + 1) % NUM_STAGES
                offs_k += BLOCK_SIZE_K
        curAccMbar = txl.get_buffer(mbar_tmem, bufIdxAccMbar)
        txl.mbar_wait(curAccMbar, phaseAccMbar)
        c = txl.tmem_load(curAcc)
        bufIdxAccMbar = (bufIdxAccMbar + 1) % 2
        if bufIdxAccMbar == 0:
            phaseAccMbar = phaseAccMbar ^ 1
        newBufIdxAccMbar = (newBufIdxAccMbar + 1) % 2
        acc = tl.reshape(c, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        accx, accy = tl.split(acc)
        c0 = accx.to(dtype)
        c_desc.store([offs_am, offs_bn], c0)
        c1 = accy.to(dtype)
        c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
        bufIdxAcc = (bufIdxAcc + 1) % 2


def matmul_persistent_nows_tma_txl_bw4(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_nows_tma_txl_bw_kernel4[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 4,
                "EPILOGUE_SUBTILE": True,
            },
            num_stages=4,
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode="ttgir")
def matmul_persistent_nows_tma_txl_bw_kernel5(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a0 = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b0 = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    out_buf = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], dtype=dtype, num_stages=1
    )
    out = txl.get_buffer(out_buf, 0)
    acc0 = txl.tmem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=2)
    zeros = txl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxP = 0
    bufIdxC = 0
    phase = 0
    mbar_tmem = txl.mbar_alloc(1, num_stages=2)
    bufIdxAcc = 0
    bufIdxAccMbar = 0
    phaseAccMbar = 0
    newBufIdxAccMbar = 1
    tid = txl.tid(0)
    producer_tile_id = start_pid
    producer_pid_m, producer_pid_n = _compute_pid(
        producer_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
    )
    producer_offs_am = producer_pid_m * BLOCK_SIZE_M
    producer_offs_bn = producer_pid_n * BLOCK_SIZE_N
    offs_k = 0
    for i in tl.static_range(NUM_STAGES):
        cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
        a = txl.get_buffer(a0, bufIdxP)
        b = txl.get_buffer(b0, bufIdxP)
        txl.mbar_expect(
            cur_mbar, BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2
        )
        txl.tma_load(a, a_desc, [producer_offs_am, offs_k], cur_mbar)
        txl.tma_load(b, b_desc, [producer_offs_bn, offs_k], cur_mbar)
        bufIdxP = (bufIdxP + 1) % NUM_STAGES
        offs_k += BLOCK_SIZE_K
        if offs_k == K:
            producer_tile_id += NUM_SMS
            producer_pid_m, producer_pid_n = _compute_pid(
                producer_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
            )
            producer_offs_am = producer_pid_m * BLOCK_SIZE_M
            producer_offs_bn = producer_pid_n * BLOCK_SIZE_N
            offs_k = 0
    cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
    a = txl.get_buffer(a0, bufIdxC)
    b = txl.get_buffer(b0, bufIdxC)
    txl.mbar_wait(cur_mbar, phase)
    curAcc = txl.get_buffer(acc0, bufIdxAcc)
    prevAcc = curAcc
    txl.tmem_store(curAcc, zeros)
    curAcc1 = txl.get_buffer(acc0, 1)
    txl.tmem_store(curAcc1, zeros)
    curAccMbar = txl.get_buffer(mbar_tmem, bufIdxAccMbar)
    curAcc = txl.dotx(a, b.T, curAcc)
    bufIdxC = (bufIdxC + 1) % NUM_STAGES
    if bufIdxC == 0:
        phase = phase ^ 1
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_total_tiles = (num_tiles - start_pid + NUM_SMS - 1) // NUM_SMS * num_k
    tile_id = start_pid
    cnt_iter = 1
    pid_m, pid_n = _compute_pid(
        tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
    )
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    for k in tl.range(1, num_total_tiles):
        cur_mbar = txl.get_buffer(mbar_producer, bufIdxC)
        newAccMbar = txl.get_buffer(mbar_tmem, newBufIdxAccMbar)
        curAccMbar = txl.get_buffer(mbar_tmem, bufIdxAccMbar)
        a = txl.get_buffer(a0, bufIdxC)
        b = txl.get_buffer(b0, bufIdxC)
        txl.mbar_wait(cur_mbar, phase)
        useD = cnt_iter != 0
        curAcc = txl.dotx(a, b.T, curAcc)
        txl.mbar_wait(curAccMbar, phaseAccMbar)
        if cnt_iter == 0:
            c = txl.tmem_load(prevAcc)
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(c, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                accx, accy = tl.split(acc)
                c0 = accx.to(dtype)
                txl.smem_store(out, c0)
                txl.tma_store(out, c_desc, [offs_am, offs_bn])
                c1 = accy.to(dtype)
                txl.tma_store_wait(0)
                txl.smem_store(out, c1)
                txl.tma_store(out, c_desc, [offs_am, offs_bn + BLOCK_SIZE_N // 2])
            else:
                c_desc.store([offs_am, offs_bn], c)
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
        bufIdxC = (bufIdxC + 1) % NUM_STAGES
        if bufIdxC == 0:
            phase = phase ^ 1
        bufIdxAccMbar = (bufIdxAccMbar + 1) % 2
        if bufIdxAccMbar == 0:
            phaseAccMbar = phaseAccMbar ^ 1
        newBufIdxAccMbar = (newBufIdxAccMbar + 1) % 2
        if k < num_total_tiles - NUM_STAGES + 1:
            cur_mbar = txl.get_buffer(mbar_producer, bufIdxP)
            a = txl.get_buffer(a0, bufIdxP)
            b = txl.get_buffer(b0, bufIdxP)
            txl.mbar_expect(
                cur_mbar,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
            txl.tma_load(a, a_desc, [producer_offs_am, offs_k], cur_mbar)
            txl.tma_load(b, b_desc, [producer_offs_bn, offs_k], cur_mbar)
            bufIdxP = (bufIdxP + 1) % NUM_STAGES
            offs_k += BLOCK_SIZE_K
            if offs_k == K:
                producer_tile_id += NUM_SMS
                producer_pid_m, producer_pid_n = _compute_pid(
                    producer_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
                )
                producer_offs_am = producer_pid_m * BLOCK_SIZE_M
                producer_offs_bn = producer_pid_n * BLOCK_SIZE_N
                offs_k = 0
        cnt_iter += 1
        if cnt_iter == num_k:
            cnt_iter = 0
            tile_id += NUM_SMS
            pid_m, pid_n = _compute_pid(
                tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
            )
            prevAcc = curAcc
            bufIdxAcc = (bufIdxAcc + 1) % 2
            curAcc = txl.get_buffer(acc0, bufIdxAcc)
    curAccMbar = txl.get_buffer(mbar_tmem, bufIdxAccMbar)
    txl.mbar_wait(curAccMbar, phaseAccMbar)
    c = txl.tmem_load(prevAcc)
    if EPILOGUE_SUBTILE:
        acc = tl.reshape(c, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        accx, accy = tl.split(acc)
        c0 = accx.to(dtype)
        txl.smem_store(out, c0)
        txl.tma_store(out, c_desc, [offs_am, offs_bn])
        c1 = accy.to(dtype)
        txl.tma_store_wait(0)
        txl.smem_store(out, c1)
        txl.tma_store(out, c_desc, [offs_am, offs_bn + BLOCK_SIZE_N // 2])
    else:
        c_desc.store([offs_am, offs_bn], c)


def matmul_persistent_nows_tma_txl_bw5(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_nows_tma_txl_bw_kernel5[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 4,
                "EPILOGUE_SUBTILE": True,
            },
            num_stages=4,
            num_warps=8,
            num_warpgroups=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_tma_txl_bw_kernel6(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tid = txl.tid(0)
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    a_buf = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    b_buf = txl.smem_alloc(
        [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES
    )
    acc_buf = txl.tmem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, num_stages=2
    )
    zeros = txl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    out_buf = txl.smem_alloc(
        [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], dtype=dtype, num_stages=1
    )
    out = txl.get_buffer(out_buf, 0)
    mbar_producer_buf = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_consumer_buf = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxP = 0
    bufIdxC = 0
    phase = 0
    mbar_acc_producer_buf = txl.mbar_alloc(1, num_stages=2)
    mbar_acc_consumer_buf = txl.mbar_alloc(1, num_stages=2)
    bufIdxAccP = 0
    bufIdxAccC = 0
    phaseAcc = 0
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_total_tiles = (num_tiles - start_pid + NUM_SMS - 1) // NUM_SMS * num_k
    tile_id = start_pid
    pid_m, pid_n = _compute_pid(
        tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
    )
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    if txl.is_warp([0, 1, 2, 3, 4, 5, 6, 7]):
        k = 0
        for i in tl.range(0, num_total_tiles):
            k += 1
            if k == num_k:
                k = 0
                acc = txl.get_buffer(acc_buf, bufIdxAccC)
                acc_reg = txl.tmem_load(acc)
                mbar_acc_producer = txl.get_buffer(mbar_acc_producer_buf, bufIdxAccC)
                mbar_acc_consumer = txl.get_buffer(mbar_acc_consumer_buf, bufIdxAccC)
                txl.mbar_wait(mbar_acc_producer, phaseAcc)
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
                )
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N
                tile_id += NUM_SMS
                acc_split = tl.reshape(acc_reg, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc_split = tl.permute(acc_split, (0, 2, 1))
                accx, accy = tl.split(acc_split)
                c0 = accx.to(dtype)
                txl.smem_store(out, c0)
                txl.tma_store(out, c_desc, [offs_am, offs_bn])
                c1 = accy.to(dtype)
                txl.tma_store_wait(0)
                txl.smem_store(out, c1)
                txl.tma_store(out, c_desc, [offs_am, offs_bn + BLOCK_SIZE_N // 2])
                txl.mbar_arrive(mbar_acc_consumer, tid == 0)
                bufIdxAccC += 1
                if bufIdxAccC == 2:
                    bufIdxAccC = 0
                    phaseAcc ^= 1
        txl.tma_store_wait(0)
    elif txl.is_warp([8]):
        k = 0
        phase = 0
        phaseAcc = 1
        for i in tl.range(0, num_total_tiles):
            mbar_producer = txl.get_buffer(mbar_producer_buf, bufIdxC)
            mbar_consumer = txl.get_buffer(mbar_consumer_buf, bufIdxC)
            a = txl.get_buffer(a_buf, bufIdxC)
            b = txl.get_buffer(b_buf, bufIdxC)
            txl.mbar_wait(mbar_producer, phase)
            acc = txl.get_buffer(acc_buf, bufIdxAccP)
            mbar_acc_producer = txl.get_buffer(mbar_acc_producer_buf, bufIdxAccP)
            useD = k != 0
            mbars = [mbar_consumer, mbar_acc_producer]
            mbarPreds = [True, k == num_k - 1]
            acc = txl.dotx(a, b.T, acc, useD, mbars=mbars, mbarPreds=mbarPreds)
            bufIdxC += 1
            if bufIdxC == NUM_STAGES:
                bufIdxC = 0
                phase ^= 1
            k += 1
            if k == num_k:
                k = 0
                bufIdxAccP += 1
                if bufIdxAccP == 2:
                    bufIdxAccP = 0
                    phaseAcc ^= 1
                mbar_acc_consumer = txl.get_buffer(mbar_acc_consumer_buf, bufIdxAccP)
                txl.mbar_wait(mbar_acc_consumer, phaseAcc)
    elif txl.is_warp([9]):
        k = 0
        phase = 1
        for i in tl.range(0, num_total_tiles):
            if k == 0:
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
                )
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N
                tile_id += NUM_SMS
            offs_k = k * BLOCK_SIZE_K
            mbar_consumer = txl.get_buffer(mbar_consumer_buf, bufIdxP)
            txl.mbar_wait(mbar_consumer, phase)
            mbar_producer = txl.get_buffer(mbar_producer_buf, bufIdxP)
            txl.mbar_expect(
                mbar_producer,
                BLOCK_SIZE_M * BLOCK_SIZE_K * 2 + BLOCK_SIZE_N * BLOCK_SIZE_K * 2,
            )
            a = txl.get_buffer(a_buf, bufIdxP)
            b = txl.get_buffer(b_buf, bufIdxP)
            txl.tma_load(a, a_desc, [offs_am, offs_k], mbar_producer)
            txl.tma_load(b, b_desc, [offs_bn, offs_k], mbar_producer)
            bufIdxP += 1
            if bufIdxP == NUM_STAGES:
                bufIdxP = 0
                phase ^= 1
            k += 1
            if k == num_k:
                k = 0


def matmul_persistent_tma_txl_bw6(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_persistent_tma_txl_bw_kernel6[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=False,
    )
    return c


def cublas_matmul(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(
        f"cublas [M={M}, N={N}, K={K}]",
        {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2.0 * M * N * K},
    ):
        cublas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(
        f"torch [M={M}, N={N}, K={K}]",
        {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2.0 * M * N * K},
    ):
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


def bench(K, dtype, reps=100, warmup_reps=25, algo="hopper_triton_ws_persistent"):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    bn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = bn.T.contiguous()
    if cublas is not None:
        bench_fn("cublas", reps, warmup_reps, cublas_matmul, a, b)
    if algo == "hopper_triton_ws_persistent":
        bench_fn("tma_persistent", reps, warmup_reps, matmul_tma_persistent, a, b)
    elif algo == "hopper_txl_naive":
        bench_fn("naive_tma_txl", reps, warmup_reps, matmul_naive_tma_txl, a, b)
    elif algo == "hopper_txl_ws_persistent":
        bench_fn(
            "tma_ws_persistent_txl",
            reps,
            warmup_reps,
            matmul_tma_ws_persistent_txl,
            a,
            b,
        )
    elif algo == "blackwell_txl_ws_persistent":
        bench_fn(
            "blackwell_ws_tma_persistent",
            reps,
            warmup_reps,
            matmul_persistent_tma_txl_bw6,
            a,
            b,
        )
    return
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]
    for ws in warp_specialize:
        ws_str = "_ws" if ws else ""
        if HAS_TENSOR_DESC and ws:
            bench_fn(
                f"descriptor_persistent{ws_str}",
                reps,
                warmup_reps,
                lambda a, b: matmul_descriptor_persistent(a, b, ws),
                a,
                b,
            )


def run_test(expect, fn, a, b, label, enabled=True, log=False):
    print(f"  {label}: ...", end="")
    if enabled:
        actual = fn(a, b)
        if log:
            print()
            print(expect[0][:128])
            print(actual[0][:128])
            avgx = (expect - actual).mean(dim=0)
            avgy = (expect - actual).mean(dim=1)
            print(avgx)
            print(avgx.size())
            print((avgx == 0.0).sum().item())
            print(avgy)
            print(avgy.size())
            print((avgy == 0.0).sum().item())
        passed = torch.allclose(expect, actual.to(expect.dtype), atol=1.0)
        icon = "✅" if passed else "❌"
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")


def validate(M, N, K, dtype, log=False, algo="hopper_triton_ws_persistent"):
    print(f"M={M!r}, N={N!r}, K={K!r}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    bn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = bn.T.contiguous()
    naive_result = cublas_matmul(a, b).to(torch.float16)
    if algo == "hopper_triton_ws_persistent":
        run_test(
            naive_result,
            lambda a, b: matmul_tma_persistent(a, b),
            a,
            b,
            "TMA Original Persistent",
            log=True,
        )
    elif algo == "hopper_txl_naive":
        run_test(
            naive_result,
            lambda a, b: matmul_naive_tma_txl(a, b),
            a,
            b,
            "TXL TMA Naive",
            log=log,
        )
    elif algo == "hopper_txl_ws_persistent":
        run_test(
            naive_result,
            lambda a, b: matmul_tma_ws_persistent_txl(a, b),
            a,
            b,
            "TXL TMA WS Persistent",
            log=True,
        )
    elif algo == "blackwell_txl_ws_persistent":
        run_test(
            naive_result,
            lambda a, b: matmul_persistent_tma_txl_bw6(a, b),
            a,
            b,
            "Blackwell ws tma persistent matmul",
            log=True,
        )
    return
    kernels = []
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]
    for (kernel, label, enabled), warp_specialize in itertools.product(
        kernels, warp_specialize
    ):
        label = f"{label} (warp_specialize={warp_specialize})"
        skipped = (
            is_hopper() and warp_specialize and (kernel != matmul_descriptor_persistent)
        )
        enabled = enabled and (not warp_specialize or HAS_TENSOR_DESC) and (not skipped)
        run_test(
            naive_result,
            lambda a, b: kernel(a, b, warp_specialize),
            a,
            b,
            label,
            enabled,
        )
    print()


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = []
    if precision == "fp8":
        metric_names = ["tflop8/s"] + metric_names
    elif precision == "fp16":
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


def profile(M, N, K, dtype, log=False):
    print(f"M={M!r}, N={N!r}, K={K!r}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    print(matmul_tma_ws_persistent_txl(a, b))


def test_matmul(dump_dir=None, algo="hopper_triton_ws_persistent"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()
    import os

    print(os.getpid())
    from triton import knobs

    if algo.startswith("blackwell"):
        knobs.runtime.override_arch = "sm100"
    knobs.autotuning.print = True
    knobs.compilation.always_compile = True
    if dump_dir:
        knobs.compilation.dump_ir = True
        knobs.cache.dump_dir = dump_dir
    if args.prec == "fp8" and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
    else:
        dtype = torch.float8_e4m3fn if args.prec == "fp8" else torch.float16
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1
        torch.manual_seed(0)
        validate(128 * 16, 256 * 16, args.K_range[0], dtype, log=True, algo=algo)
        proton.start("matmul", hook="triton")
        for K in range(10, 11):
            bench(2**K, dtype, algo=algo)
        proton.finalize()
        show_profile(args.prec, "matmul")


if __name__ == "__main__":
    test_matmul("dump/mm0211", "hopper_txl_ws_persistent")
