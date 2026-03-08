"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import os
import triton
import triton.language as tl
import time

try:
    import teraxlang as txl

    Has_TXL = True
    from triton.tools.tensor_descriptor import TensorDescriptor

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    import triton.profiler.language as pl
    import triton.profiler as proton
    from teraxlang.language.semantic import TXLSemantic

    pl.enable_semantic("triton")
    pl.enable_semantic_obj(TXLSemantic)
    print("TXL")
except Exception as e:
    print(e)

    class txl:
        class Config:
            def __init__(
                self, config, num_stages=1, num_warps=1, num_warpgroups=1, pre_hook=None
            ):
                pass

        @staticmethod
        def jit(
            use_txl=False, diff_mode="ttir", diff_select=-1, log_dir="", src_file=""
        ):

            def decorator(func):
                return func

            return decorator

        @staticmethod
        def autotune(configs=[], key=""):

            def decorator(func):
                return func

            return decorator

    Has_TXL = False
    DEVICE = torch.device("cuda:0")
    print("No txl")


def is_hip():
    return False


def is_cuda():
    return True


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _host_descriptor_pre_hook(nargs):
    NUM_CONSUMER_GROUPS = nargs.get("NUM_CONSUMERS", 1)
    BLOCK_M = nargs["BLOCK_M"] // NUM_CONSUMER_GROUPS
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]
configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
    )
    for BM in [64, 128]
    for BN in [64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    configs = [
        triton.Config(
            dict(BLOCK_M=64, BLOCK_N=64),
            num_stages=2,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and (conf.num_warps == 8)
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = (0, start_m * BLOCK_M)
    elif STAGE == 2:
        lo, hi = (start_m * BLOCK_M, (start_m + 1) * BLOCK_M)
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetkv_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1000000.0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = desc_v.load([offsetkv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetkv_y += BLOCK_N
    return (acc, l_i, m_i)


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = desc_q.load([qo_offset_y, 0])
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return (dk, dv)


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    start_m = start_n
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=True,
    )
    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=False,
    )
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)
    m = m[:, None]
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,
        MASK=True,
    )
    end_n -= num_steps * MASK_BLOCK_N2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,
        MASK=False,
    )
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


tma_nows_best_config = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "NUM_STAGES": 3,
    "NUM_CONSUMERS": 1,
}
tma_ws_best_config = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "NUM_CONSUMERS": 2,
    "NUM_STAGES": 2,
}
tma_ws_best_config_split = {
    "BLOCK_M": 256,
    "BLOCK_N": 64,
    "NUM_CONSUMERS": 2,
    "NUM_STAGES": 2,
}


@txl.autotune(
    configs=[
        txl.Config(
            tma_nows_best_config,
            num_stages=3,
            num_warps=4,
            num_warpgroups=1,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_tma_txl(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    bQ = txl.smem_alloc([BLOCK_M, HEAD_DIM], dtype=dtype)
    pMbar_bQ = txl.mbar_alloc(1)
    bQ0 = txl.get_buffer(bQ, 0)
    pMbar_bQ0 = txl.get_buffer(pMbar_bQ, 0)
    txl.mbar_expect(pMbar_bQ0, BLOCK_M * HEAD_DIM * 2)
    txl.tma_load(bQ0, desc_q, [qo_offset_y, 0], pMbar_bQ0)
    txl.mbar_wait(pMbar_bQ0, 0)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxW = 0
    bufIdxR = 0
    phase = 0
    for i in tl.static_range(1, NUM_STAGES):
        cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
        cur_bK = txl.get_buffer(bK, bufIdxW)
        cur_bV = txl.get_buffer(bV, bufIdxW)
        txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
        txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
        txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
        txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
        bufIdxW = (bufIdxW + 1) % NUM_STAGES
        offsetkv_y += BLOCK_N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
        cur_bK = txl.get_buffer(bK, bufIdxR)
        txl.mbar_wait(cur_mbar_bK, phase)
        qk = tl.dot(txl.get_buffer(bQ, 0), cur_bK.T)
        txl.dot_wait(0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
        cur_bV = txl.get_buffer(bV, bufIdxR)
        txl.mbar_wait(cur_mbar_bV, phase)
        p = p.to(dtype)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(1)
        m_i = m_ij
        bufIdxR = (bufIdxR + 1) % NUM_STAGES
        if bufIdxR == 0:
            phase = phase ^ 1
        if start_n < hi - (NUM_STAGES - 1) * BLOCK_N:
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            offsetkv_y += BLOCK_N
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl1(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxR = 0
        phase = 0
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
            cur_bK = txl.get_buffer(bK, bufIdxR)
            txl.mbar_wait(cur_mbar_bK, phase)
            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxR)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxR)
                qk = tl.dot(bQ0i, cur_bK.T)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)
            else:
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxR)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxR)
                qk = tl.dot(bQ1i, cur_bK.T)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
            cur_bV = txl.get_buffer(bV, bufIdxR)
            txl.mbar_wait(cur_mbar_bV, phase)
            p = p.to(dtype)
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(0)
            m_i = m_ij
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase = phase ^ 1
            txl.mbar_arrive(cur_mbar_PV)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


class NamedBarrier:
    WG1 = tl.constexpr(8)
    WG2 = tl.constexpr(9)


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl2(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    WG1_BAR = 8
    WG2_BAR = 9
    WG_NUM_THREADS = 128 * 2
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
            txl.bar_wait(8, 256)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
            txl.bar_wait(9, 256)
        bufIdxR = 0
        phase = 0
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
            cur_bK = txl.get_buffer(bK, bufIdxR)
            txl.mbar_wait(cur_mbar_bK, phase)
            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxR)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxR)
                qk = tl.dot(bQ0i, cur_bK.T)
                txl.bar_arrive(9, 256)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)
            else:
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxR)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxR)
                qk = tl.dot(bQ1i, cur_bK.T)
                txl.bar_arrive(8, 256)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
            cur_bV = txl.get_buffer(bV, bufIdxR)
            txl.mbar_wait(cur_mbar_bV, phase)
            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256)
            else:
                txl.bar_wait(9, 256)
            p = p.to(dtype)
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(0)
            m_i = m_ij
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase = phase ^ 1
            txl.mbar_arrive(cur_mbar_PV)
        if txl.is_warpgroup([1]):
            txl.bar_arrive(9, 256)
        else:
            txl.bar_arrive(8, 256)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


@txl.jit
def softmax_txl(m_i, l_i, qk, qk_scale, dtype):
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    p = p.to(dtype)
    return (m_i, l_i, p, alpha)


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl3(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        txl.reg_dealloc(24)
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        txl.reg_alloc(240)
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)
        if txl.is_warpgroup([1]):
            cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
            qk = tl.dot(bQ0i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        else:
            cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
            qk = tl.dot(bQ1i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        p = p.to(dtype)
        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)
            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256)
            if txl.is_warpgroup([2]):
                txl.bar_wait(9, 256)
            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                qk = tl.dot(bQ0i, cur_bK.T)
            else:
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                qk = tl.dot(bQ1i, cur_bK.T)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(1)
            txl.mbar_arrive(cur_mbar_QK)
            if txl.is_warpgroup([1]):
                txl.bar_arrive(9, 256)
            else:
                txl.bar_arrive(8, 256)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            p = p.to(dtype)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_PV)
            acc = acc * alpha[:, None]
            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl4(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)
        if txl.is_warpgroup([1]):
            cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
            qk = tl.dot(bQ0i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        else:
            cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
            qk = tl.dot(bQ1i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        p = p.to(dtype)
        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            pl.enter_scope("waitK")
            txl.mbar_wait(cur_mbar_bK, phaseK)
            pl.exit_scope("waitK")
            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256)
            if txl.is_warpgroup([2]):
                txl.bar_wait(9, 256)
            pl.enter_scope("QK")
            pl.enter_scope("QKi")
            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                qk = tl.dot(bQ0i, cur_bK.T)
            else:
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                qk = tl.dot(bQ1i, cur_bK.T)
            pl.exit_scope("QKi")
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            pl.enter_scope("waitV")
            txl.mbar_wait(cur_mbar_bV, phaseV)
            pl.exit_scope("waitV")
            pl.enter_scope("PV")
            pl.enter_scope("PVi")
            acc = tl.dot(p, cur_bV, acc)
            pl.exit_scope("PVi")
            txl.dot_wait(1)
            pl.exit_scope("QK")
            txl.mbar_arrive(cur_mbar_QK)
            if txl.is_warpgroup([1]):
                txl.bar_arrive(9, 256)
            else:
                txl.bar_arrive(8, 256)
            pl.enter_scope("SM")
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            p = p.to(dtype)
            pl.exit_scope("SM")
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_PV)
            pl.exit_scope("PV")
            pl.enter_scope("rescale")
            acc = acc * alpha[:, None]
            pl.exit_scope("rescale")
            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl5(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxR = 0
        phase = 0
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
            cur_bK = txl.get_buffer(bK, bufIdxR)
            txl.mbar_wait(cur_mbar_bK, phase)
            with pl.scope("QK"):
                if txl.is_warpgroup([1]):
                    cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxR)
                    cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxR)
                    qk = tl.dot(bQ0i, cur_bK.T)
                    txl.dot_wait(0)
                    txl.mbar_arrive(cur_mbar_QK)
                else:
                    cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxR)
                    cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxR)
                    qk = tl.dot(bQ1i, cur_bK.T)
                    txl.dot_wait(0)
                    txl.mbar_arrive(cur_mbar_QK)
            with pl.scope("Softmax"):
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
            cur_bV = txl.get_buffer(bV, bufIdxR)
            txl.mbar_wait(cur_mbar_bV, phase)
            p = p.to(dtype)
            with pl.scope("PV"):
                acc = tl.dot(p, cur_bV, acc)
                txl.dot_wait(0)
            m_i = m_ij
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase = phase ^ 1
            txl.mbar_arrive(cur_mbar_PV)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


class NamedBarrier:
    WG1 = tl.constexpr(8)
    WG2 = tl.constexpr(9)


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl6(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    WG1_BAR = 8
    WG2_BAR = 9
    WG_NUM_THREADS = 128 * 2
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            with pl.scope("waitQK12"):
                txl.mbar_wait(cur_mbar_QK1, phase)
                txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            with pl.scope("waitPV12"):
                txl.mbar_wait(cur_mbar_PV1, phase)
                txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
            txl.bar_wait(8, 256)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
            txl.bar_wait(9, 256)
        bufIdxR = 0
        phase = 0
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
            cur_bK = txl.get_buffer(bK, bufIdxR)
            with pl.scope("waitK"):
                txl.mbar_wait(cur_mbar_bK, phase)
            with pl.scope("QK"):
                if txl.is_warpgroup([1]):
                    cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxR)
                    cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxR)
                    qk = tl.dot(bQ0i, cur_bK.T)
                    txl.bar_arrive(9, 256)
                    txl.dot_wait(0)
                    txl.mbar_arrive(cur_mbar_QK)
                else:
                    cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxR)
                    cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxR)
                    qk = tl.dot(bQ1i, cur_bK.T)
                    txl.bar_arrive(8, 256)
                    txl.dot_wait(0)
                    txl.mbar_arrive(cur_mbar_QK)
            cur_max = tl.max(qk, 1)
            m_ij = tl.maximum(m_i, cur_max * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p = p.to(dtype)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
            cur_bV = txl.get_buffer(bV, bufIdxR)
            with pl.scope("waitV"):
                txl.mbar_wait(cur_mbar_bV, phase)
            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256)
            else:
                txl.bar_wait(9, 256)
            with pl.scope("PV"):
                acc = tl.dot(p, cur_bV, acc)
                txl.dot_wait(0)
            m_i = m_ij
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase = phase ^ 1
            txl.mbar_arrive(cur_mbar_PV)
        if txl.is_warpgroup([1]):
            txl.bar_arrive(9, 256)
        else:
            txl.bar_arrive(8, 256)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl7(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        txl.reg_dealloc(24)
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        txl.reg_alloc(240)
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)
        if txl.is_warpgroup([1]):
            cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
            qk = tl.dot(bQ0i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        else:
            cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
            qk = tl.dot(bQ1i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        p = p.to(dtype)
        bufIdxRK = 0
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)
            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256)
            if txl.is_warpgroup([2]):
                txl.bar_wait(9, 256)
            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                qk = tl.dot(bQ0i, cur_bK.T)
            else:
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                qk = tl.dot(bQ1i, cur_bK.T)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(1)
            txl.mbar_arrive(cur_mbar_QK)
            if txl.is_warpgroup([1]):
                txl.bar_arrive(9, 256)
            else:
                txl.bar_arrive(8, 256)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            p = p.to(dtype)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_PV)
            acc = acc * alpha[:, None]
            phaseK = phaseK ^ 1
            phaseV = phaseV ^ 1
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl_own1(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
    USE_PROFILE: tl.constexpr = True,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW)
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)
        if txl.is_warpgroup([1]):
            cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
            qk = tl.dot(bQ0i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        else:
            cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
            qk = tl.dot(bQ1i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_ij
        p = p.to(dtype)
        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)
            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256)
            if txl.is_warpgroup([2]):
                txl.bar_wait(9, 256)
            if USE_PROFILE:
                pl.enter_scope("QK")
            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                qk = tl.dot(bQ0i, cur_bK.T)
            else:
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                qk = tl.dot(bQ1i, cur_bK.T)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)
            if USE_PROFILE:
                pl.enter_scope("PV")
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(1)
            if USE_PROFILE:
                pl.exit_scope("QK")
            txl.mbar_arrive(cur_mbar_QK)
            if txl.is_warpgroup([1]):
                txl.bar_arrive(9, 256)
            if txl.is_warpgroup([2]):
                txl.bar_arrive(8, 256)
            if USE_PROFILE:
                pl.enter_scope("SM")
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            p = p.to(dtype)
            if USE_PROFILE:
                pl.exit_scope("SM")
            txl.dot_wait(0)
            if USE_PROFILE:
                pl.exit_scope("PV")
            txl.mbar_arrive(cur_mbar_PV)
            acc = acc * alpha[:, None]
            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl_tawa(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    cMbar_PV = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    if txl.is_warpgroup([0]):
        txl.reg_dealloc(40)
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y + BLOCK_M // 2, 0], pMbar_bQ1i)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxW)
            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxW)
            txl.mbar_wait(cur_mbar_QK, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    if txl.is_warpgroup([1, 2]):
        txl.reg_alloc(232)
        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
        m_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        if txl.is_warpgroup([1]):
            pMbar_bQx = pMbar_bQ0
            bQx = bQ0
        else:
            pMbar_bQx = pMbar_bQ1
            bQx = bQ1
        bQi = txl.get_buffer(bQx, 0)
        pMbar_bQi = txl.get_buffer(pMbar_bQx, 0)
        txl.mbar_wait(pMbar_bQi, 0)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)
        cur_mbar_QK = txl.get_buffer(cMbar_QK, 0)
        qk = tl.dot(bQi, cur_bK.T)
        txl.dot_wait(0)
        txl.mbar_arrive(cur_mbar_QK)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        p = p.to(dtype)
        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            acc = acc * alpha[:, None]
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)
            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxRK)
            qk = tl.dot(bQi, cur_bK.T)
            txl.dot_wait(0)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)
            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxRV)
            acc = tl.dot(p, cur_bV, acc)
            txl.mbar_arrive(cur_mbar_QK)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            p = p.to(dtype)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_PV)
            m_i = m_ij
            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1
        acc = acc * alpha[:, None]
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        m_i += tl.math.log2(l_i)
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        acc = acc / l_i[:, None]
        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc.to(dtype))


def _host_descriptor_pre_hook2(nargs):
    NUM_CONSUMER_GROUPS = nargs.get("NUM_CONSUMERS", 1)
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook2,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl_tawa2(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ = txl.smem_alloc([BLOCK_M, HEAD_DIM], dtype=dtype)
    pMbar_bQ = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    cMbar_PV = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo
    wgid = txl.warpgroup_id()
    if wgid == 0:
        txl.reg_dealloc(40)
        bQi = txl.get_buffer(bQ, 0)
        pMbar_bQi = txl.get_buffer(pMbar_bQ, 0)
        txl.mbar_expect(pMbar_bQi, BLOCK_M * HEAD_DIM * 2)
        txl.tma_load(bQi, desc_q, [qo_offset_y, 0], pMbar_bQi)
        bufIdxW = 0
        phase = 1
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)
            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxW)
            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxW)
            txl.mbar_wait(cur_mbar_QK, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)
            txl.mbar_wait(cur_mbar_PV, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)
            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase ^ 1
    else:
        txl.reg_alloc(232)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504
        bQi = txl.get_buffer(bQ, 0)
        pMbar_bQi = txl.get_buffer(pMbar_bQ, 0)
        txl.mbar_wait(pMbar_bQi, 0)
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)
        cur_mbar_QK = txl.get_buffer(cMbar_QK, 0)
        qk = tl.dot(bQi, cur_bK.T)
        txl.dot_wait(0)
        txl.mbar_arrive(cur_mbar_QK)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        p = p.to(dtype)
        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)
            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxRK)
            qk = tl.dot(bQi, cur_bK.T)
            txl.dot_wait(0)
            acc = acc * alpha[:, None]
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)
            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxRV)
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            p = p.to(dtype)
            txl.mbar_arrive(cur_mbar_PV)
            m_i = m_ij
            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1
        acc = acc * alpha[:, None]
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        desc_o.store([qo_offset_y, 0], acc.to(dtype))


@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=_host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl_test(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    o_ptr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    bQ0 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M // 2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    cMbar_QK = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    cMbar_PV = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    lo, hi = (0, N_CTX)
    offsetkv_y = offset_y + lo


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, algo=0, no_tune=False, profiling=False):
        HEAD_DIM_Q, HEAD_DIM_K = (q.shape[-1], k.shape[-1])
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        if supports_host_descriptor():
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_v = TensorDescriptor(
                v,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        ctx.grid = grid
        algo_map = {
            0: _attn_fwd_tma_txl,
            "hopper_txl_ws_naive": _attn_fwd_ws_tma_txl1,
            "hopper_txl_ws_pingpong": _attn_fwd_ws_tma_txl2,
            "hopper_txl_ws_fa3": _attn_fwd_ws_tma_txl3,
            4: _attn_fwd_ws_tma_txl4,
            5: _attn_fwd_ws_tma_txl_tawa,
            6: _attn_fwd_ws_tma_txl_tawa2,
            7: _attn_fwd_ws_tma_txl_test,
            8: _attn_fwd_ws_tma_txl5,
            9: _attn_fwd_ws_tma_txl6,
            10: _attn_fwd_ws_tma_txl_own1,
            11: _attn_fwd_ws_tma_txl7,
        }
        if profiling:
            proton.start(
                "fa3",
                backend="instrumentation",
                mode="default:sampling_strategy=selective:sampling_options=0,4,8",
                data="trace",
            )
        # Support both string keys and number keys
        kernel = algo_map.get(algo, algo_map.get(str(algo)))
        kernel[grid](
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=False,
            **extra_kern_args,
        )
        if profiling:
            proton.finalize()
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = (4, 5)
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = (32, 128, 128, 32)
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta, BATCH, N_HEAD, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            N_HEAD,
            N_CTX,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return (dq, dk, dv, None, None, None, None)


def test_attention(q, k, v, causal, sm_scale, algo=0, no_tune=False, profiling=False):
    HEAD_DIM_Q, HEAD_DIM_K = (q.shape[-1], k.shape[-1])
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    extra_kern_args = {}
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    if supports_host_descriptor():
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_o = TensorDescriptor(
            o,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
    else:
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    algo_map = {
        0: _attn_fwd_tma_txl,
        "hopper_txl_ws_naive"   : _attn_fwd_ws_tma_txl1,
        "hopper_txl_ws_pingpong": _attn_fwd_ws_tma_txl2,
        "hopper_txl_ws_fa3"     : _attn_fwd_ws_tma_txl3,
        4: _attn_fwd_ws_tma_txl4,
        5: _attn_fwd_ws_tma_txl_tawa,
        6: _attn_fwd_ws_tma_txl_tawa2,
        7: _attn_fwd_ws_tma_txl_test,
        8: _attn_fwd_ws_tma_txl5,
        9: _attn_fwd_ws_tma_txl6,
        10: _attn_fwd_ws_tma_txl_own1,
        11: _attn_fwd_ws_tma_txl7,
    }
    if profiling:
        proton.start(
            "fa3",
            backend="instrumentation",
            mode="default:sampling_strategy=selective:sampling_options=0,4,8",
            data="trace",
        )
    algo_map[algo][grid](
        sm_scale,
        M,
        q.shape[0],
        q.shape[1],
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        FP8_OUTPUT=q.dtype == torch.float8_e5m2,
        STAGE=stage,
        warp_specialize=False,
        **extra_kern_args,
    )
    if profiling:
        proton.finalize()
    return o


attention = test_attention
import sys
from teraxlang.tutorials.utils.test_util import attention_ref
import math

try:
    from teraxlang.tests.flash_attn.cute.interface import flash_attn_func

    PYFLASH = True
    HAS_FLASH = True
    print("Has Flash")
except Exception as e:
    HAS_FLASH = False
    print("Has No Flash")
if Has_TXL:
    HAS_FLASH = False
    print("TXL over Flash")


def test_op(
    Z,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    dtype=torch.float16,
    algo=0,
    no_tune=False,
    profiling=False,
):
    q = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    q1 = q.permute(0, 2, 1, 3).contiguous()
    k1 = k.permute(0, 2, 1, 3).contiguous()
    v1 = v.permute(0, 2, 1, 3).contiguous()
    test_outs = []
    if HAS_FLASH:
        if PYFLASH:
            flash_out, lse = flash_attn_func(q1, k1, v1, causal=causal)
            flash_out = flash_out.half()
        else:
            flash_out = flash_attn_func(q1, k1, v1, causal=causal).half()
        flash_out = flash_out.permute(0, 2, 1, 3).contiguous()
        test_outs.append(flash_out)
    elif Has_TXL:
        sm_scale = 1 / math.sqrt(HEAD_DIM)
        txl_out = attention(q, k, v, causal, sm_scale, algo, no_tune, profiling).half()
        test_outs.append(txl_out)
    if profiling:
        exit()
    ref_out, ref_attn = attention_ref(q1, k1, v1, causal=causal)
    ref_out = ref_out.permute(0, 2, 1, 3).contiguous()
    for actual_out in test_outs:
        print(f"Output max diff: {(actual_out - ref_out).abs().max().item()}")
        print(f"Output mean diff: {(actual_out - ref_out).abs().mean().item()}")
        assert torch.allclose(ref_out, actual_out, atol=0.01, rtol=0)


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
BATCH, N_HEADS, HEAD_DIM = (16, 32, 128)
TORCH_HAS_FP8 = False
configs = []
for mode in ["fwd"]:
    for causal in [False]:
        for warp_specialize in [False, True] if is_blackwell() else [False]:
            if mode == "bwd" and (not causal):
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(14, 15)],
                    line_arg="provider",
                    line_vals=(["triton-fp16"] if Has_TXL else [])
                    + (["triton-fp8"] if TORCH_HAS_FP8 else [])
                    + (["flash"] if HAS_FLASH else []),
                    line_names=(["Triton [FP16]"] if Has_TXL else [])
                    + (["Triton [FP8]"] if TORCH_HAS_FP8 else [])
                    + (["Flash-3"] if HAS_FLASH else []),
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                )
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    mode,
    provider,
    device=DEVICE,
    algo=0,
    no_tune=False,
):
    time.sleep(1)
    BATCH = int(16384 / N_CTX)
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    q = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False
    )
    print(q.shape)
    k = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False
    )
    v = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False
    )
    if "triton" in provider:
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1 / math.sqrt(HEAD_DIM)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, algo, no_tune)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
    if provider == "flash":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
    ms = triton.testing.do_bench(fn, warmup=25, rep=1000)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5
    print(flops_per_matmul)
    return total_flops / ms * 1e-09


def run_test(algo=0, dump_dir=None):
    import os
    from triton import knobs

    knobs.autotuning.print = True
    knobs.compilation.always_compile = True
    if dump_dir:
        knobs.compilation.dump_ir = True
        knobs.cache.dump_dir = dump_dir
    no_tune = True
    print("TEST...")
    PROFILING = False
    test_op(
        16,
        32,
        1024,
        128,
        False,
        dtype=torch.float16,
        algo=algo,
        no_tune=no_tune,
        profiling=PROFILING,
    )
    print("BENCH...")
    bench_flash_attention.run(
        save_path=".", print_data=True, algo=algo, no_tune=no_tune
    )


if __name__ == "__main__":
    run_test(3, dump_dir="dump/fa0212")
