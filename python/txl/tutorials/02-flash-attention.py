"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

#import pytest
import torch
import os

import triton
import triton.language as tl

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
        def jit(use_txl=False, diff_mode='ttir', diff_select=-1, log_dir='', src_file=''):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def autotune(configs=[], key=''):
            def decorator(func):
                return func
            return decorator
    Has_TXL = False
    DEVICE = torch.device('cuda:0')
    print("No txl")


def is_hip():
    #return triton.runtime.driver.active.get_current_target().backend == "hip"
    return False


def is_cuda():
    #return triton.runtime.driver.active.get_current_target().backend == "cuda"
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
    FP8_OUTPUT = nargs.get("FP8_OUTPUT", False)
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if FP8_OUTPUT:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
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
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=64, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

###################################################
# Fwd
###################################################

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetkv_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([offsetkv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetkv_y += BLOCK_N
    return acc, l_i, m_i


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))

###################################################
# Done Fwd
###################################################



###################################################
# Bwd
###################################################

@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)

###################################################
# Done Bwd
###################################################

###################################################
# TXL Fwd
###################################################
# DEPRECATE notma
#notma_best_config = {'BLOCK_M':64, 'BLOCK_N':32, 'num_stages': 3, 'num_warps':4}
tma_nows_best_config = {'BLOCK_M':64, 'BLOCK_N':64, 'NUM_STAGES': 3, 'NUM_CONSUMERS': 1} #stages 3, num warps 4
tma_ws_best_config = {'BLOCK_M':128, 'BLOCK_N':128, 'NUM_CONSUMERS': 2, 'NUM_STAGES': 2} # stages: 3, num warps: 4, num_warpgroups: 3

###################################################
# TXL + TMA + multi-stage
###################################################

@txl.autotune(
    configs=[
        txl.Config(
            tma_nows_best_config,
            num_stages=3,
            num_warps=4,
            num_warpgroups=1,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
@txl.jit
def _attn_fwd_tma_txl(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])
    bQ = txl.smem_alloc([BLOCK_M, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ = txl.mbar_alloc(1) # 1 buffer
    bQ0 = txl.get_buffer(bQ, 0)
    pMbar_bQ0 = txl.get_buffer(pMbar_bQ, 0)
    txl.mbar_expect(pMbar_bQ0, BLOCK_M * HEAD_DIM * 2)
    txl.tma_load(bQ0, desc_q, [qo_offset_y, 0], pMbar_bQ0)
    txl.mbar_wait(pMbar_bQ0, 0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bufIdxW = 0
    bufIdxR = 0
    phase = 0

    for i in tl.static_range(1, NUM_STAGES):
        #TODO: ctx len boundary check
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

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
        cur_bK = txl.get_buffer(bK, bufIdxR)
        txl.mbar_wait(cur_mbar_bK, phase)

        qk = tl.dot(txl.get_buffer(bQ, 0), cur_bK.T)
        txl.dot_wait(0)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]

        # load v
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
        cur_bV = txl.get_buffer(bV, bufIdxR)
        txl.mbar_wait(cur_mbar_bV, phase)

        # update acc
        p = p.to(dtype)

        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(1)

        # update m_i and l_i
        m_i = m_ij

        bufIdxR = (bufIdxR + 1) % NUM_STAGES
        if bufIdxR == 0:
            phase = phase ^ 1

        if start_n < hi-(NUM_STAGES-1)*BLOCK_N:
            #TODO: ctx len boundary check
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

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


###################################################
# TXL + TMA + FA3 Algo1
###################################################

@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
@txl.jit
#@txl.jit(diff_mode="llir")
def _attn_fwd_ws_tma_txl1(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])
    bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, N_CTX
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
        txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)

        bufIdxW = 0 # write buffer
        phase = 1

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)

            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW) # wait for the same buffer
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)

            # TODO: tma_expect_and_load
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
                phase = phase^1



    if txl.is_warpgroup([1, 2]):

        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
        # initialize pointer to m and l
        # These are in regs
        m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)

        bufIdxR = 0 # read buffer
        phase = 0
        # loop over k, v and update accumulator
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # -- compute qk ----
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
            cur_bK = txl.get_buffer(bK, bufIdxR)
            txl.mbar_wait(cur_mbar_bK, phase)

            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxR) # wait for the same buffer
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxR)
                qk = tl.dot(bQ0i, cur_bK.T)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)
            else: # [2]
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxR)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxR)
                qk = tl.dot(bQ1i, cur_bK.T)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]

            # load v
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
            cur_bV = txl.get_buffer(bV, bufIdxR)
            txl.mbar_wait(cur_mbar_bV, phase)

            # update acc
            p = p.to(dtype)

            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(0)

            # update m_i and l_i
            m_i = m_ij

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase = phase ^ 1

            txl.mbar_arrive(cur_mbar_PV)

        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)

        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))


###################################################
# TXL + TMA + FA3 Algo2 Pingpong
###################################################

class NamedBarrier:
    WG1 = tl.constexpr(8)
    WG2 = tl.constexpr(9)

#test_config = {'BLOCK_M':64, 'BLOCK_N':256, 'NUM_CONSUMERS': 2, 'NUM_STAGES': 2} # stages: 3, num warps: 4, num_warpgroups: 3
@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
@txl.jit
def _attn_fwd_ws_tma_txl2(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    #off_z = tl.program_id(2)
    #off_h = tl.program_id(1)
    #off_hz = off_z * H + off_h

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])
    bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
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

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo


    if txl.is_warpgroup([0]):
        #txl.reg_dealloc(24)

        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)

        bufIdxW = 0 # write buffer
        phase = 1

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)

            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW) # wait for the same buffer
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)

            # TODO: tma_expect_and_load
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
                phase = phase^1



    if txl.is_warpgroup([1, 2]):
        #txl.reg_alloc(240)

        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)

            # first let wg1 to start
            #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
        # initialize pointer to m and l
        # These are in regs
        m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
            # WG1 just start
            #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
            txl.bar_wait(8, 256)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
            # WG2 start after wg1 gemm0
            #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)
            txl.bar_wait(9, 256)

        bufIdxR = 0 # read buffer
        phase = 0
        # loop over k, v and update accumulator
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # -- compute qk ----
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxR)
            cur_bK = txl.get_buffer(bK, bufIdxR)
            txl.mbar_wait(cur_mbar_bK, phase)

            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxR) # wait for the same buffer
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxR)
                qk = tl.dot(bQ0i, cur_bK.T)

                # TODO whether before dot wait?
                #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                txl.bar_arrive(9, 256)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)

            else: # [2]
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxR)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxR)
                qk = tl.dot(bQ1i, cur_bK.T)

                # TODO whether before dot wait?
                #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                txl.bar_arrive(8, 256)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mbar_QK)


            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]

            # load v
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxR)
            cur_bV = txl.get_buffer(bV, bufIdxR)
            txl.mbar_wait(cur_mbar_bV, phase)

            # Downgrade if put before load v
            if txl.is_warpgroup([1]):
                #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                txl.bar_wait(8, 256)
            else:
                #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)
                txl.bar_wait(9, 256)


            # update acc
            p = p.to(dtype)

            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(0)

            # update m_i and l_i
            m_i = m_ij

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase = phase ^ 1

            txl.mbar_arrive(cur_mbar_PV)

        if txl.is_warpgroup([1]):
            #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
            txl.bar_arrive(9, 256)
        else:
            #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
            txl.bar_arrive(8, 256)

        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)

        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))


###################################################
# TXL + TMA + FA3 Algo3 Pingpong + Intra Overlap
###################################################

@txl.jit
def softmax_txl(m_i, l_i, qk, qk_scale, dtype):
    # -- compute softamx, block arg updates ----
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]

    # udpate p
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    # update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    m_i = m_ij

    # update acc, NOTE: p position is important
    p = p.to(dtype)
    return m_i, l_i, p, alpha

@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
#@txl.jit(src_file='dump/BFTFLZXGNHJ5H24JXMORYIYU7P6KTCTQ5YJ5EZJSF77MVLWKGJNA/_attn_fwd_ws_tma_txl3.ptx')
@txl.jit
#@txl.jit(diff_mode='ttgir', diff_select=6, log_dir='dump')
def _attn_fwd_ws_tma_txl3(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])
    bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    #WG1_BAR = 8
    #WG2_BAR = 9
    #WG_NUM_THREADS = 128 * 2

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, N_CTX
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
        txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)

        bufIdxW = 0 # write buffer
        phase = 1

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)

            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW) # wait for the same buffer
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)

            # TODO: tma_expect_and_load
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
                phase = phase^1


    if txl.is_warpgroup([1, 2]):
        txl.reg_alloc(240)

        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)

            # first let wg1 to start
            #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)

        # initialize pointer to m and l
        # These are in regs
        m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

        ## load and wait Q
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
            # WG1 just start
            #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
            # WG2 start after wg1 gemm0
            #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)


        # -- prologue --

        # TODO: write in txl.jit for reuse
        ## load and wait K
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)

        if txl.is_warpgroup([1]):
            cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
            qk = tl.dot(bQ0i, cur_bK.T)

            # TODO whether before dot wait?
            #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)

        else: # [2]
            cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
            qk = tl.dot(bQ1i, cur_bK.T)

            # TODO whether before dot wait?
            #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        #acc = acc * alpha[:, None]

        # update m_i and l_i
        m_i = m_ij

        # update acc
        p = p.to(dtype)

        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0

        # pass: p, l_i, m_i, acc
        # loop over k, v and update accumulator
        for start_n in range(lo+BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # -- load k ----
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)

            # Now only consider gemm 0 and softmax(gemm 1)
            # --- wait to start gemm 1 ---
            # case 1: wg1 earlys start
            # case 2: wait the release from last iter gemm 0 ends?
            if txl.is_warpgroup([1]):
                # WG1 just start
                #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                txl.bar_wait(8, 256)
            if txl.is_warpgroup([2]):
                # WG2 start after wg1 gemm0
                #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)
                txl.bar_wait(9, 256)

            if txl.is_warpgroup([1]):
                cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                qk = tl.dot(bQ0i, cur_bK.T)

            else: # [2]
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                qk = tl.dot(bQ1i, cur_bK.T)

            # -- compute pv j-1 ----
            # load v
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)

            ## Downgrade if put before load v
            #if txl.is_warpgroup([1]):
            #    txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
            #else:
            #    txl.bar_wait(WG2_BAR, WG_NUM_THREADS)

            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = tl.dot(p, cur_bV, acc)

            # TODO: before or after wait? oh previously is also before QK wait
            if txl.is_warpgroup([1]):
                #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                txl.bar_arrive(9, 256)
            else:
                #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                txl.bar_arrive(8, 256)
            txl.dot_wait(1)
            # --- release QK finished ---
            txl.mbar_arrive(cur_mbar_QK)

            #m_i, l_i, p, alpha = softmax_txl(m_i, l_i, qk, qk_scale, dtype)

            # -- compute softamx, block arg updates ----
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            # udpate p
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

            # update output accumulator
            txl.dot_wait(0)
            # --- release PV j-1 finished ---
            txl.mbar_arrive(cur_mbar_PV)

            acc = acc * alpha[:, None]

            # update acc, NOTE: p position is important
            p = p.to(dtype)

            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1


        #if txl.is_warpgroup([1]):
        #    txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
        #else:
        #    txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)

        # -- last iter --
        # load v
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        #if txl.is_warpgroup([1]):
        #    cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
        #else:
        #    cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)

        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        #txl.mbar_arrive(cur_mbar_PV)

        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)

        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))

###################################################
# PROFILING: TXL + TMA + FA3 Algo3 Pingpong + Intra Overlap
###################################################

@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl4(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):

    with pl.scope("kernel"):
        dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
        byte_count: tl.constexpr = 2 if dtype == tl.float16 else 1
        tl.static_assert(BLOCK_N <= HEAD_DIM)
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        y_dim = Z * H * N_CTX
        # If no host desc, then make device desc
        desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_M, HEAD_DIM])
        if FP8_OUTPUT: #v_shape = (BATCH, H, HEAD_DIM, N_CTX)
            y_dim_v = Z * H * HEAD_DIM
            desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim_v, N_CTX], strides=[N_CTX, 1],
                                            block_shape=[HEAD_DIM, BLOCK_N])
        else:
            desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                            block_shape=[BLOCK_N, HEAD_DIM])
        desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
        desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_M, HEAD_DIM])

        offset_y = off_z * (N_CTX * H) + off_h * N_CTX
        qo_offset_y = offset_y + start_m * BLOCK_M


        # load q: it will stay in SRAM throughout
        #q = desc_q.load([qo_offset_y, 0])
        bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
        pMbar_bQ0 = txl.mbar_alloc(1)
        bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
        pMbar_bQ1 = txl.mbar_alloc(1)

        bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
        if FP8_OUTPUT:
            bV = txl.smem_alloc([HEAD_DIM, BLOCK_N], dtype=dtype, num_stages=NUM_STAGES)
        else:
            bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
        pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
        pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

        cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
        cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
        cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
        cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

        #WG1_BAR = 8
        #WG2_BAR = 9
        #WG_NUM_THREADS = 128 * 2

        # TODO: func type mismatch

        # range of values handled by this stage
        lo, hi = 0, N_CTX
        offsetkv_y = offset_y + lo


        if txl.is_warpgroup([0]):
            with pl.scope("wg0"):
                bQ0i = txl.get_buffer(bQ0, 0)
                pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
                bQ1i = txl.get_buffer(bQ1, 0)
                pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

                with pl.scope("waitQ"):
                    txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * byte_count)
                    txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
                    txl.mbar_wait(pMbar_bQ0i, 0)
                    txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * byte_count)
                    txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
                    txl.mbar_wait(pMbar_bQ1i, 0)

                bufIdxW = 0 # write buffer
                phase = 1

                for start_n in range(lo, hi, BLOCK_N):
                    start_n = tl.multiple_of(start_n, BLOCK_N)
                    cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
                    cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
                    cur_bK = txl.get_buffer(bK, bufIdxW)
                    cur_bV = txl.get_buffer(bV, bufIdxW)

                    cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW) # wait for the same buffer
                    cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
                    cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
                    cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)

                    # TODO: tma_expect_and_load
                    with pl.scope("waitQK"):
                        txl.mbar_wait(cur_mbar_QK1, phase)
                        txl.mbar_wait(cur_mbar_QK2, phase)
                    txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * byte_count)
                    txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)

                    with pl.scope("waitPV"):
                        txl.mbar_wait(cur_mbar_PV1, phase)
                        txl.mbar_wait(cur_mbar_PV2, phase)
                    txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * byte_count)
                    if FP8_OUTPUT:
                        txl.tma_load(cur_bV, desc_v, [off_hz * HEAD_DIM, start_n], cur_mbar_bV)
                    else:
                        txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)

                    offsetkv_y += BLOCK_N
                    bufIdxW = (bufIdxW + 1) % NUM_STAGES
                    if bufIdxW == 0:
                        phase = phase^1


        if txl.is_warpgroup([1, 2]):
            with pl.scope("wg12"):

                if txl.is_warpgroup([1]):
                    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)

                    # first let wg1 to start
                    #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                    txl.bar_arrive(8, 256)
                else:
                    offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
                # if txl.tid(0) == 129:
                #     txl.print('here1')
                # initialize pointer to m and l
                # These are in regs
                m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
                # load scales
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                ## load and wait Q
                bQ0i = txl.get_buffer(bQ0, 0)
                pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
                bQ1i = txl.get_buffer(bQ1, 0)
                pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

                with pl.scope("waitQ init"):
                    if txl.is_warpgroup([1]):
                        txl.mbar_wait(pMbar_bQ0i, 0)
                        # WG1 just start
                        #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                    if txl.is_warpgroup([2]):
                        txl.mbar_wait(pMbar_bQ1i, 0)
                        # WG2 start after wg1 gemm0
                        #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)


                    # -- prologue --

                    # TODO: write in txl.jit for reuse
                    ## load and wait K
                    cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
                    cur_bK = txl.get_buffer(bK, 0)
                    txl.mbar_wait(cur_mbar_bK, 0)

                    if txl.is_warpgroup([1]):
                        cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
                        qk = tl.dot(bQ0i, cur_bK.T)

                        # TODO whether before dot wait?
                        #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                        txl.dot_wait(0)
                        txl.mbar_arrive(cur_mbar_QK)

                    else: # [2]
                        cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
                        qk = tl.dot(bQ1i, cur_bK.T)

                        # TODO whether before dot wait?
                        #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                        txl.dot_wait(0)
                        txl.mbar_arrive(cur_mbar_QK)

                    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, None]

                    p = tl.math.exp2(qk)
                    l_ij = tl.sum(p, 1)
                    # -- update m_i and l_i
                    alpha = tl.math.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    # -- update output accumulator --
                    #acc = acc * alpha[:, None]

                    # update m_i and l_i
                    m_i = m_ij

                    # update acc
                    p = p.to(dtype)

                bufIdxRK = 1
                bufIdxRV = 0
                phaseK = 0
                phaseV = 0

                # pass: p, l_i, m_i, acc
                # loop over k, v and update accumulator
                for start_n in range(lo+BLOCK_N, hi, BLOCK_N):
                    start_n = tl.multiple_of(start_n, BLOCK_N)

                    # -- load k ----
                    cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
                    cur_bK = txl.get_buffer(bK, bufIdxRK)
                    with pl.scope("waitK"):
                        txl.mbar_wait(cur_mbar_bK, phaseK)

                    # Now only consider gemm 0 and softmax(gemm 1)
                    # --- wait to start gemm 1 ---
                    # case 1: wg1 earlys start
                    # case 2: wait the release from last iter gemm 0 ends?
                    if txl.is_warpgroup([1]):
                        # WG1 just start
                        #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                        with pl.scope("wait Kick QK"):
                            txl.bar_wait(8, 256)
                    if txl.is_warpgroup([2]):
                        # WG2 start after wg1 gemm0
                        #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)
                        with pl.scope("wait Kick QK"):
                            txl.bar_wait(9, 256)

                    if txl.is_warpgroup([1]):
                        cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                        cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                        qk = tl.dot(bQ0i, cur_bK.T)

                    else: # [2]
                        cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                        cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                        qk = tl.dot(bQ1i, cur_bK.T)

                    # -- compute pv j-1 ----
                    # load v
                    cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
                    cur_bV = txl.get_buffer(bV, bufIdxRV)
                    with pl.scope("waitV"):
                        txl.mbar_wait(cur_mbar_bV, phaseV)

                    ## Downgrade if put before load v
                    #if txl.is_warpgroup([1]):
                    #    txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                    #else:
                    #    txl.bar_wait(WG2_BAR, WG_NUM_THREADS)

                    # note that this non transposed v for FP8 is only supported on Blackwell
                    if FP8_OUTPUT:
                        acc = tl.dot(p, cur_bV.T, acc)
                    else:
                        acc = tl.dot(p, cur_bV, acc)

                    with pl.scope("dotQK"):
                        txl.dot_wait(1)
                    # TODO: before or after wait? oh previously is also before QK wait
                    if txl.is_warpgroup([1]):
                        #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                        with pl.scope("Kick QK"):
                            txl.bar_arrive(9, 256)
                    else:
                        #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                        with pl.scope("Kick QK"):
                            txl.bar_arrive(8, 256)
                    # --- release QK finished ---
                    txl.mbar_arrive(cur_mbar_QK)

                    #m_i, l_i, p, alpha = softmax_txl(m_i, l_i, qk, qk_scale, dtype)

                    with pl.scope("Softmax"):
                        # -- compute softamx, block arg updates ----
                        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                        qk = qk * qk_scale - m_ij[:, None]

                        # udpate p
                        p = tl.math.exp2(qk)
                        l_ij = tl.sum(p, 1)
                        # update m_i and l_i
                        alpha = tl.math.exp2(m_i - m_ij)
                        l_i = l_i * alpha + l_ij
                        m_i = m_ij

                        # update acc, NOTE: p position is important
                        p = p.to(dtype)

                    # update output accumulator
                    with pl.scope("dotPV"):
                        txl.dot_wait(0)
                    # --- release PV j-1 finished ---
                    txl.mbar_arrive(cur_mbar_PV)

                    acc = acc * alpha[:, None]

                    bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
                    if bufIdxRK == 0:
                        phaseK = phaseK ^ 1
                    bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
                    if bufIdxRV == 0:
                        phaseV = phaseV ^ 1


                #if txl.is_warpgroup([1]):
                #    txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                #else:
                #    txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)

                # -- last iter --
                # load v
                cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
                #if txl.is_warpgroup([1]):
                #    cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                #else:
                #    cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                with pl.scope("Epilogue"):
                    cur_bV = txl.get_buffer(bV, bufIdxRV)
                    txl.mbar_wait(cur_mbar_bV, phaseV)

                    # note that this non transposed v for FP8 is only supported on Blackwell
                    if FP8_OUTPUT:
                        acc = tl.dot(p, cur_bV.T, acc)
                    else:
                        acc = tl.dot(p, cur_bV, acc)
                    txl.dot_wait(0)
                    #txl.mbar_arrive(cur_mbar_PV)

                    # epilogue
                    m_i += tl.math.log2(l_i)
                    acc = acc / l_i[:, None]
                    m_ptrs = M + off_hz * N_CTX + offs_m
                    tl.store(m_ptrs, m_i)

                    if txl.is_warpgroup([1]):
                        desc_o.store([qo_offset_y, 0], acc.to(dtype))
                    if txl.is_warpgroup([2]):
                        desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))

@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
)
@txl.jit
def _attn_fwd_ws_tma_txl4_causal(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):

    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    byte_count: tl.constexpr = 2 if dtype == tl.float16 else 1
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT: 
        y_dim_v = Z * H * HEAD_DIM
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim_v, N_CTX], strides=[N_CTX, 1],
                                        block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) 
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    if FP8_OUTPUT:
        bV = txl.smem_alloc([HEAD_DIM, BLOCK_N], dtype=dtype, num_stages=NUM_STAGES)
    else:
        bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    lo, hi = 0, (start_m + 1) * BLOCK_M
    mask_begin = start_m * BLOCK_M
    offsetkv_y = offset_y + lo

    if txl.is_warpgroup([0]):
        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * byte_count)
        txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * byte_count)
        txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
        txl.mbar_wait(pMbar_bQ1i, 0)

        bufIdxW = 0
        phase = 1

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)

            cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW) # wait for the same buffer
            cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
            cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
            cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)

            # TODO: tma_expect_and_load
            txl.mbar_wait(cur_mbar_QK1, phase)
            txl.mbar_wait(cur_mbar_QK2, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * byte_count)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)

            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * byte_count)
            if FP8_OUTPUT:
                txl.tma_load(cur_bV, desc_v, [off_hz * HEAD_DIM, start_n], cur_mbar_bV)
            else:
                txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)

            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase^1


    if txl.is_warpgroup([1, 2]):

        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

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

        else: # [2]
            cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
            qk = tl.dot(bQ1i, cur_bK.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_QK)

        if lo >= mask_begin:
            mask = offs_m[:, None] >= (lo + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
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

        for start_n in range(lo+BLOCK_N, hi, BLOCK_N):
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

            else: # [2]
                cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                qk = tl.dot(bQ1i, cur_bK.T)

            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)

            if FP8_OUTPUT:
                acc = tl.dot(p, cur_bV.T, acc)
            else:
                acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(1)

            if txl.is_warpgroup([1]):
                txl.bar_arrive(9, 256)
            else:
                txl.bar_arrive(8, 256)

            txl.mbar_arrive(cur_mbar_QK)

            if start_n >= mask_begin:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
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

        if FP8_OUTPUT:
            acc = tl.dot(p, cur_bV.T, acc)
        else:
            acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)

        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)

        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))

###################################################
# TXL + TAWA
###################################################

@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
@txl.jit
#@txl.jit(diff_mode="llir")
#@txl.jit(src_file="dump/fa1117/3WABGMTQLV7L7LEIYIXT3C3VKRQAQBHFZ5KDTN7AZI47XB5T767Q/_attn_fwd_ws_tma_txl_tawa.ptx")
#@txl.jit(src_file="/workspace/_attn_fwd_ws_tma_txl_tawa.ptx")
def _attn_fwd_ws_tma_txl_tawa(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              #o_ptr,
              #stride_om: tl.constexpr,
              #stride_on: tl.constexpr,

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    #cur_o_ptr = ((qo_offset_y + tl.arange(0, BLOCK_M//2)) * stride_om)[:, None] + tl.arange(0, stride_om)[None, :]


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])

    bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)
    #bQ = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    #pMbar_bQ = txl.mbar_alloc(1)
    #bQi = txl.get_buffer(bQ, 0)
    #pMbar_bQi = txl.get_buffer(pMbar_bQ, 0)
    #bQ0i = txl.smem_slice(bQi, 0, BLOCK_M//2, 0)
    #bQ1i = txl.smem_slice(bQi, BLOCK_M//2, BLOCK_M//2, 0)

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    #cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    cMbar_PV = txl.mbar_alloc(256, num_stages=NUM_STAGES)

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, N_CTX
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
        txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)

        #txl.mbar_expect(pMbar_bQi, BLOCK_M * HEAD_DIM * 2)
        #txl.tma_load(bQi, desc_q, [qo_offset_y, 0], pMbar_bQi)

        bufIdxW = 0 # write buffer
        phase = 1

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)

            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxW) # wait for the same buffer
            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxW)

            # TODO: tma_expect_and_load
            txl.mbar_wait(cur_mbar_QK, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)

            txl.mbar_wait(cur_mbar_PV, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)

            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase^1

    if txl.is_warpgroup([1, 2]):
        txl.reg_alloc(232)

        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)
            #o_block_ptr = tl.make_block_ptr(
            #    base=o_ptr + offset_y,
            #    shape=(N_CTX, HEAD_DIM),
            #    strides=(stride_om, stride_on),
            #    offsets=(start_m * BLOCK_M, 0),
            #    block_shape=(BLOCK_M//2, HEAD_DIM),
            #    order=(1, 0),
            #)
            #cur_o_ptr = o_ptr+((qo_offset_y + tl.arange(0, BLOCK_M//2)) * stride_om)[:, None] + tl.arange(0, stride_om)[None, :]

        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
            #o_block_ptr = tl.make_block_ptr(
            #    base=o_ptr + offset_y,
            #    shape=(N_CTX, HEAD_DIM),
            #    strides=(stride_om, stride_on),
            #    offsets=(start_m * BLOCK_M + BLOCK_M//2, 0),
            #    block_shape=(BLOCK_M//2, HEAD_DIM),
            #    order=(1, 0),
            #)
            #cur_o_ptr = o_ptr+((qo_offset_y + tl.arange(BLOCK_M//2, BLOCK_M)) * stride_om)[:, None] + tl.arange(0, stride_om)[None, :]

        # initialize pointer to m and l
        # These are in regs
        m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)

        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        # wait Q
        if txl.is_warpgroup([1]):
            txl.mbar_wait(pMbar_bQ0i, 0)
        if txl.is_warpgroup([2]):
            txl.mbar_wait(pMbar_bQ1i, 0)
        #txl.mbar_wait(pMbar_bQi, 0)

        # -- prologue --

        # TODO: write in txl.jit for reuse

        ## load and wait K
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)

        cur_mbar_QK = txl.get_buffer(cMbar_QK, 0)

        if txl.is_warpgroup([1]):
            qk = tl.dot(bQ0i, cur_bK.T)
        else: # [2]
            qk = tl.dot(bQ1i, cur_bK.T)
        txl.dot_wait(0)
        # load new K
        txl.mbar_arrive(cur_mbar_QK)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        #acc = acc * alpha[:, None]

        # update m_i and l_i
        m_i = m_ij

        # TXL: cast reordered
        p = p.to(dtype)

        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0

        # loop over k, v and update accumulator
        for start_n in range(lo+BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # tawa: update acc
            #p = p.to(dtype)
            # tawa: rescale acc
            #acc = acc * alpha[:, None]

            # -- compute qk i+1 ----
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)

            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxRK) # wait for the same buffer

            if txl.is_warpgroup([1]):
                qk = tl.dot(bQ0i, cur_bK.T)
            else: # [2]
                qk = tl.dot(bQ1i, cur_bK.T)
            txl.dot_wait(0)

            # -- compute pv i ----
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)

            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxRV)

            # note that this non transposed v for FP8 is only supported on Blackwell
            # here we have a p layout convertion from mma to dot_op
            acc = tl.dot(p, cur_bV, acc)

            # TAWA: delayed QK arrive
            txl.mbar_arrive(cur_mbar_QK)


            # Softmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij


            # TXL: cast reordered
            p = p.to(dtype)


            # tawa : PV
            txl.dot_wait(0)
            # -- update output accumulator --

            # TXL: reordered
            acc = acc * alpha[:, None]

            # finish PV
            #txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_PV)


            # update m_i and l_i
            m_i = m_ij

            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1

        # -- last iter --
        # tawa: update acc
        #p = p.to(dtype)
        # tawa: rescale acc
        #acc = acc * alpha[:, None]

        # load v
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)

        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        #txl.mbar_arrive(cur_mbar_PV)

        # epilogue

        m_i += tl.math.log2(l_i)
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        acc = acc / l_i[:, None]

        if txl.is_warpgroup([1]):
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
            #tl.store(o_block_ptr, acc.to(dtype))
            #tl.store(cur_o_ptr, acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))
            #tl.store(cur_o_ptr, acc.to(dtype))

###################################################
# TXL + TAWA2
###################################################

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
            pre_hook = _host_descriptor_pre_hook2,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
@txl.jit
def _attn_fwd_ws_tma_txl_tawa2(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    #cur_o_ptr = ((qo_offset_y + tl.arange(0, BLOCK_M//2)) * stride_om)[:, None] + tl.arange(0, stride_om)[None, :]


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])

    #bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    #pMbar_bQ0 = txl.mbar_alloc(1)
    #bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
    #pMbar_bQ1 = txl.mbar_alloc(1)

    bQ = txl.smem_alloc([BLOCK_M, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ = txl.mbar_alloc(1)

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    #cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    cMbar_PV = txl.mbar_alloc(256, num_stages=NUM_STAGES)

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    wgid = txl.warpgroup_id()
    #if wgid == 2:
    #    o_offset_y = qo_offset_y + BLOCK_M//2
    #else:
    #    o_offset_y = qo_offset_y

    if wgid == 0:
        txl.reg_dealloc(40)

        #bQ0i = txl.get_buffer(bQ0, 0)
        #pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        #bQ1i = txl.get_buffer(bQ1, 0)
        #pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)
        bQi = txl.get_buffer(bQ, 0)
        pMbar_bQi = txl.get_buffer(pMbar_bQ, 0)

        txl.mbar_expect(pMbar_bQi, BLOCK_M * HEAD_DIM * 2)
        txl.tma_load(bQi, desc_q, [qo_offset_y, 0], pMbar_bQi)
        #txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
        #txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)

        bufIdxW = 0 # write buffer
        phase = 1

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
            cur_bK = txl.get_buffer(bK, bufIdxW)
            cur_bV = txl.get_buffer(bV, bufIdxW)

            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxW) # wait for the same buffer
            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxW)

            # TODO: tma_expect_and_load
            txl.mbar_wait(cur_mbar_QK, phase)
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)

            txl.mbar_wait(cur_mbar_PV, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
            txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)

            offsetkv_y += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase = phase^1

    else:
        txl.reg_alloc(232)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        #if wgid == 1:
        #    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)

        #else:
        #    offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)

        # initialize pointer to m and l
        # These are in regs
        #m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
        #l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
        #acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

        bQi = txl.get_buffer(bQ, 0)
        pMbar_bQi = txl.get_buffer(pMbar_bQ, 0)
        #bQ1i = txl.get_buffer(bQ1, 0)
        #pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        # wait Q
        #if txl.is_warpgroup([1]):
        #    txl.mbar_wait(pMbar_bQ0i, 0)
        #if txl.is_warpgroup([2]):
        #    txl.mbar_wait(pMbar_bQ1i, 0)
        txl.mbar_wait(pMbar_bQi, 0)

        # -- prologue --

        # TODO: write in txl.jit for reuse

        ## load and wait K
        cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
        cur_bK = txl.get_buffer(bK, 0)
        txl.mbar_wait(cur_mbar_bK, 0)

        cur_mbar_QK = txl.get_buffer(cMbar_QK, 0)

        #if txl.is_warpgroup([1]):
        #    qk = tl.dot(bQ0i, cur_bK.T)
        #else: # [2]
        #    qk = tl.dot(bQ1i, cur_bK.T)
        qk = tl.dot(bQi, cur_bK.T)
        txl.dot_wait(0)
        # load new K
        txl.mbar_arrive(cur_mbar_QK)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        #acc = acc * alpha[:, None]

        # update m_i and l_i
        m_i = m_ij

        # update acc
        p = p.to(dtype)

        bufIdxRK = 1
        bufIdxRV = 0
        phaseK = 0
        phaseV = 0

        # loop over k, v and update accumulator
        for start_n in range(lo+BLOCK_N, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # -- compute qk i+1 ----
            cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
            cur_bK = txl.get_buffer(bK, bufIdxRK)
            txl.mbar_wait(cur_mbar_bK, phaseK)

            cur_mbar_QK = txl.get_buffer(cMbar_QK, bufIdxRK) # wait for the same buffer

            #if txl.is_warpgroup([1]):
            #    qk = tl.dot(bQ0i, cur_bK.T)
            #else: # [2]
            #    qk = tl.dot(bQ1i, cur_bK.T)
            qk = tl.dot(bQi, cur_bK.T)
            txl.dot_wait(0)

            # TXL: rescale acc
            acc = acc * alpha[:, None]

            # -- compute pv i ----
            cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
            cur_bV = txl.get_buffer(bV, bufIdxRV)
            txl.mbar_wait(cur_mbar_bV, phaseV)

            cur_mbar_PV = txl.get_buffer(cMbar_PV, bufIdxRV)

            # TXL: cast reordered
            #p = p.to(dtype)

            # note that this non transposed v for FP8 is only supported on Blackwell
            # here we have a p layout convertion from mma to dot_op
            acc = tl.dot(p, cur_bV, acc)
            txl.dot_wait(0)

            # TAWA: delayed QK arrive
            txl.mbar_arrive(cur_mbar_QK)


            # Softmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --

            # TXL: reordered
            #acc = acc * alpha[:, None]
            p = p.to(dtype)

            # finish PV
            #txl.dot_wait(0)
            txl.mbar_arrive(cur_mbar_PV)


            # update m_i and l_i
            m_i = m_ij

            bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
            if bufIdxRK == 0:
                phaseK = phaseK ^ 1
            bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
            if bufIdxRV == 0:
                phaseV = phaseV ^ 1
        # -- last iter --
        acc = acc * alpha[:, None]
        # load v
        cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
        cur_bV = txl.get_buffer(bV, bufIdxRV)
        txl.mbar_wait(cur_mbar_bV, phaseV)

        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, cur_bV, acc)
        txl.dot_wait(0)
        #txl.mbar_arrive(cur_mbar_PV)

        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)

        #if txl.is_warpgroup([1]):
        #    desc_o.store([qo_offset_y, 0], acc.to(dtype))
        #if txl.is_warpgroup([2]):
        #    desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))
        desc_o.store([qo_offset_y, 0], acc.to(dtype))

@txl.autotune(
    configs=[
        txl.Config(
            tma_ws_best_config,
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook = _host_descriptor_pre_hook,
        )
    ],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
 )
@txl.jit
def _attn_fwd_ws_tma_txl_test(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              o_ptr,
              stride_om: tl.constexpr,
              stride_on: tl.constexpr,

              # NOTE: txl
              NUM_STAGES: tl.constexpr,  #
              NUM_CONSUMERS: tl.constexpr  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # If no host desc, then make device desc
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    #cur_o_ptr = ((qo_offset_y + tl.arange(0, BLOCK_M//2)) * stride_om)[:, None] + tl.arange(0, stride_om)[None, :]


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])

    bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)

    bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    #cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    #cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK = txl.mbar_alloc(256, num_stages=NUM_STAGES)
    cMbar_PV = txl.mbar_alloc(256, num_stages=NUM_STAGES)

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, algo=0, no_tune=False, profiling=False):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}


        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if supports_host_descriptor():
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dtype = q.dtype

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if dtype == torch.float8_e5m2:
                n_ctx = v.shape[3]
                y_dim_v = v.shape[0] * v.shape[1] * v.shape[2]
                desc_v = TensorDescriptor(v, shape=[y_dim_v, n_ctx], strides=[n_ctx, 1], block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
            #return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[1], q.shape[0])

        ctx.grid = grid
        #if no_tune:
        #    if algo == 0:
        #        extra_kern_args.update(tma_nows_best_config)
        #    else:
        #        extra_kern_args.update(tma_ws_best_config)
        algo_map = {
            0: _attn_fwd_tma_txl,
            1: _attn_fwd_ws_tma_txl1,
            2: _attn_fwd_ws_tma_txl2,
            3: _attn_fwd_ws_tma_txl3,
            4: _attn_fwd_ws_tma_txl4,
            5: _attn_fwd_ws_tma_txl4_causal,
            # 5: _attn_fwd_ws_tma_txl_tawa,
            # 6: _attn_fwd_ws_tma_txl_tawa2,
            # 7: _attn_fwd_ws_tma_txl_test,
        }

        if profiling:
            proton.start("fa3", backend="instrumentation", mode='default:sampling_strategy=selective:sampling_options=0,4,8', data="trace")
        algo_map[algo][grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #

            #o_ptr=o,
            #stride_om=o.stride(-2),
            #stride_on=o.stride(-1),

            warp_specialize=False,  #
            **extra_kern_args)

        if profiling:
            proton.finalize()

        #ctx.save_for_backward(q, k, v, o, M)
        #ctx.sm_scale = sm_scale
        #ctx.HEAD_DIM = HEAD_DIM_K
        #ctx.causal = causal
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
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        return dq, dk, dv, None, None, None, None


attention = _attention.apply


###################################################
### Helpers
###################################################

# Validate results
import sys
#sys.path.insert(0, './tools')  # Insert at beginning
from txl.tests.test_util import attention_ref
import math

try:
    #from flash_attn.flash_attn_interface import \
    #    flash_attn_qkvpacked_func as flash_attn_func
    # fa3
    from txl.tests.flash_attn.cute.interface import flash_attn_func
    PYFLASH = True

    HAS_FLASH = True

    print("Has Flash")
except Exception as e:
    #import pdb;pdb.set_trace()
    HAS_FLASH = False
    print("Has No Flash")


#if HAS_FLASH:
#    Has_TXL = False
#    print("Flash over TXL")
if Has_TXL:
    HAS_FLASH = False
    print("TXL over Flash")

#@pytest.mark.parametrize("Z", [1, 4])
#@pytest.mark.parametrize("H", [2, 48])
#@pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
#@pytest.mark.parametrize("HEAD_DIM", [64, 128])
#@pytest.mark.parametrize("causal", [True])  # FIXME: Non-causal tests do not pass at the moment.
#@pytest.mark.parametrize("warp_specialize", [False, True] if is_blackwell() else [False])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16, algo=0, no_tune=False, profiling=False):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE).normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE).normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE).normal_(mean=0.0, std=0.5))
    q1 = q.permute(0,2,1,3).contiguous()
    k1 = k.permute(0,2,1,3).contiguous()
    v1 = v.permute(0,2,1,3).contiguous()
    if dtype == torch.float8_e5m2:
        v = v.permute(0,1,3,2).contiguous().to(torch.float8_e5m2)
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
    print(f"q sample: {q[0,0,0,:8]}")
    print(f"k sample: {k[0,0,0,:8]}")
    print(f"v sample: {v[0,0,0,:8]}")
    test_outs = []
    # txl
    if HAS_FLASH:
        #tri_out = flash_attn_func(q1, k1, v1, softmax_scale=sm_scale, causal=causal).half()
        if PYFLASH:
            flash_out, lse = flash_attn_func(q1, k1, v1, causal=causal)
            flash_out = flash_out.half()
        else:
            flash_out = flash_attn_func(q1, k1, v1, causal=causal).half()
        flash_out = flash_out.permute(0,2,1,3).contiguous()
        test_outs.append(flash_out)
    elif Has_TXL:
        txl_out = attention(q, k, v, causal, 1/math.sqrt(HEAD_DIM), algo, no_tune, profiling).half()
        test_outs.append(txl_out)

    if profiling:
        exit()

    ## OLD
    #dout = torch.randn_like(q)
    ## reference implementation
    #M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    #p = torch.matmul(q, k.transpose(2, 3)) * sm_scale # this causes the result of tri attention to non-0s, in TMA mode
    #if causal:
    #    p[:, :, M == 0] = float("-inf")
    #p = torch.softmax(p.float(), dim=-1).half()
    #ref_out = torch.matmul(p, v)
    ref_out, ref_attn = attention_ref(q1, k1, v1, causal=causal)
    ref_out = ref_out.permute(0,2,1,3).contiguous()
    #print(ref_out)
    #print(tri_out)

    # compare
    #NOTE:
    # TMA will produce 0s if not having other kernels executed (like torch.matmul)
    # non-TMA is more accurate than TMA version, but under 4x32x1024x64, both are not fufilling the allclose
    # this correctness issue only happens when get too many B and H (maybe S also)

    ## OLD
    for actual_out in test_outs:
        if dtype == torch.float8_e5m2:
            actual_out = actual_out.to(torch.float16)
        print(actual_out.shape)
        print(f"Output max diff: {(actual_out - ref_out).abs().max().item()}")
        print(f"Output mean diff: {(actual_out - ref_out).abs().mean().item()}")
        print(f"actual sample: {actual_out[0,0,0,:8]}")
        print(f"ref sample:    {ref_out[0,0,0,:8]}")
        assert torch.allclose(ref_out, actual_out, atol=1e-2, rtol=0)

    #rtol = 0.0
    ## Relative tolerance workaround for known hardware limitation of MI200 GPU.
    ## For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    #if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
    #    rtol = 1e-2


TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128

TORCH_HAS_FP8 = True
TORCH_HAS_FP16 = True
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False, True]:
        for warp_specialize in [False, True] if is_blackwell() else [False]:
            if mode == "bwd" and not causal:
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    # x_vals=[2**i for i in range(14, 15)],
                    line_arg="provider",
                    line_vals=(["triton-fp16"] if TORCH_HAS_FP16 else []) + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                    (["flash"] if HAS_FLASH else []),
                    line_names=(["Triton [FP16]"] if TORCH_HAS_FP16 else [])  + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                    (["Flash-3"] if HAS_FLASH else []),
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="TFLOPS",
                    plot_name=
                    f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE, no_tune=False):
    # follow fa3 paper
    BATCH = int(16384 / N_CTX)
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    # q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    # k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    # v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    if "triton" in provider:
        if mode == "fwd" and "fp8" in provider:
            v_shape = (BATCH, H, HEAD_DIM, N_CTX)
        else:
            v_shape = (BATCH, H, N_CTX, HEAD_DIM)
        q = torch.randn(
                (BATCH, H, N_CTX, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        k = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(v_shape, dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1/math.sqrt(HEAD_DIM)
        if causal:
            fn = lambda: attention(q, k, v, causal, sm_scale, 5, no_tune)
        else:
            fn = lambda: attention(q, k, v, causal, sm_scale, 4, no_tune)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=1000, rep=1000)

    if provider == "flash":
        # qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        # fn = lambda: flash_attn_func(qkv, causal=causal)
        q = q.permute(0,2,1,3).contiguous()
        k = k.permute(0,2,1,3).contiguous()
        v = v.permute(0,2,1,3).contiguous()
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=1000, rep=1000)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


def run_test(algo=0, dump_dir=None):
    from triton import knobs

    knobs.autotuning.print=True
    knobs.compilation.always_compile=True
    
    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir

    # only works on post-Ampere GPUs right now
    no_tune=True # has best config
    #no_tune=False # no best config

    print("TEST...")
    #test_op(1, 2, 1024, 128, False, dtype=torch.float16, no_tune=no_tune)

    PROFILING=False
    #test_op(16, 32, 1024, 128, False, dtype=torch.float16, algo=0, no_tune=no_tune, profiling=PROFILING)
    #test_op(16, 32, 1024, 128, False, dtype=torch.float16, algo=1, no_tune=no_tune, profiling=PROFILING)
    #test_op(16, 32, 1024, 128, False, dtype=torch.float16, algo=2, no_tune=no_tune, profiling=PROFILING)
    # test_op(16, 32, 1024, 128, False, dtype=torch.float8_e5m2, algo=4, no_tune=no_tune, profiling=PROFILING)

    # torch.float8_e5m2 + causal may cause large numerical error, compare with tawa output the kernel is correct
    # only algo5 is causal
    # test_op(16, 32, 1024, 128, True, dtype=torch.float8_e5m2, algo=5, no_tune=no_tune, profiling=PROFILING)
    # test_op(16, 32, 1024, 128, True, dtype=torch.float16, algo=5, no_tune=no_tune, profiling=PROFILING)

    #test_op(1, 2, 1536, 128, False, dtype=torch.float16, algo=4, no_tune=no_tune, profiling=PROFILING)
    # test_op(16, 32, 1024, 128, False, dtype=torch.float16, algo=algo, no_tune=no_tune, profiling=PROFILING)

    print("BENCH...")
    bench_flash_attention.run(save_path=".", print_data=True, no_tune=no_tune)

if __name__ == "__main__":
    #run_test(6, dump_dir='dump/fa1113')
    #run_test(5, dump_dir='dump/fa1117')
    run_test()

