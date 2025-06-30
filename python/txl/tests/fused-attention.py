"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

import txl

# ENABLE_LHS_TO_TMEM is an experimental environment variable for Blackwell.
# If it is set to 1 it can improve performance of Blackwell attention. However,
# it defaults to 0 as it is known to cause correctness issues outside of the
# _attn_fwd_tma kernel below.

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)
#HAS_TMA_DESC = False

if HAS_TMA_DESC:
    print("TMA benchmarks will be running with experimental grid constant TMA descriptor.", )
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.", )


# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


# Assume BLOCK_M is multiple of BLOCK_N
@txl.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr,
                    #offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # STAGE 1 and 2 are for causal
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M # context with multiple of BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M # remaining context
        lo = txl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = txl.next_subtensor(K_block_ptr, (0, lo)) # D, B
    V_block_ptr = txl.next_subtensor(V_block_ptr, (lo, 0)) # N, D
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = txl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = txl.load(K_block_ptr)
        qk = txl.dot(q, k)
        if STAGE == 2:
            offs_m = start_m * BLOCK_M + txl.arange(0, BLOCK_M)
            offs_n = start_n + txl.arange(0, BLOCK_N)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = qk * qk_scale + txl.where(mask, 0, -1.0e6)
            m_ij = txl.maximum(m_i, txl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = txl.maximum(m_i, txl.max(qk, 1) * qk_scale) # max then scale
            qk = qk * qk_scale - m_ij[:, None] # BM, BN
        p = txl.exp2(qk)
        l_ij = txl.sum(p, 1)
        # -- update m_i and l_i
        alpha = txl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij # BM
        # -- update output accumulator --
        acc = acc * alpha[:, None] # BM, BN
        # update acc
        v = txl.load(V_block_ptr)
        if fp8_v:
            p = p.to(txl.float8e5)
        else:
            p = p.to(txl.float16) # 32 to 16
        acc = txl.dot(p, v, acc) # (BM, BN) * (BN, D)
        # update m_i and l_i
        m_i = m_ij
        K_block_ptr = txl.next_subtensor(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = txl.next_subtensor(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@txl.jit
def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                        desc_k, desc_v,  #
                        offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                        N_CTX: tl.constexpr):
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
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl._experimental_descriptor_load(desc_k, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype).T
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
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl._experimental_descriptor_load(desc_v, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype)
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        offsetkv_y += BLOCK_N
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.

#BLOCK_M: 64, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 4
#BLOCK_M: 64, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 4, num_buffers_warp_spec: 1, num_consumer_groups: 1
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, 
        num_stages=s, 
        num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]

#configs = [
#    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, }, 
#        #num_consumer_groups=2,
#        #num_buffers_warp_spec=3,
#        num_consumer_groups=1,
#        num_buffers_warp_spec=1,
#        num_stages=s, 
#        num_warps=w) \
#    for BM in [128, ]\
#    for BN in [64]\
#    for s in ([1] if is_hip() else [1])\
#    for w in [4]\
#]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
#@txl.jit(diff_mode='ttgir')
@txl.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              Q_layout, K_layout, V_layout, O_layout, M_layout,
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              ):
    txl.static_assert(BLOCK_N <= HEAD_DIM)

    #start_m = tl.program_id(0)
    #off_hz = tl.program_id(1)
    start_m = txl.bidx.x()
    off_hz = txl.bidx.y()

    off_z = off_hz // H
    off_h = off_hz % H



    Q_block_ptr = txl.subtensor(
        Q,
        (off_z, off_h, start_m*BLOCK_M),
        Q_layout,
        block_shape=(BLOCK_M, HEAD_DIM),
    )
    V_block_ptr = txl.subtensor(
        V,
        (off_z, off_h),
        V_layout,
        block_shape=(BLOCK_N, HEAD_DIM),
    )
    K_block_ptr = txl.subtensor(
        K,
        (off_z, off_h),
        K_layout,
        block_shape=(HEAD_DIM, BLOCK_N),
    )
    O_block_ptr = txl.subtensor(
        Out,
        (off_z, off_h, start_m * BLOCK_M),
        O_layout,
        block_shape=(BLOCK_M, HEAD_DIM),
    )
    M_block_ptr = txl.subtensor(
        M,
        (off_z, off_h, start_m * BLOCK_M),
        M_layout,
        block_shape=(BLOCK_M, ),
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = txl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = txl.ones([BLOCK_M], dtype=tl.float32)
    acc = txl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = txl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        #if txl.thread0():
        #    txl.print("Stage1")
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE,
                                        #offs_m, offs_n,
                                        N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        #if txl.thread0():
        #    txl.print("Stage2")
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2,
                                        #offs_m, offs_n,
                                        N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += txl.log2(l_i)
    acc = acc / l_i[:, None]
    #m_ptrs = M + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    #tl.store(m_ptrs, m_i)
    txl.store(M_block_ptr, m_i)
    txl.store(O_block_ptr, acc.to(Out.type.element_ty))


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs_tma = [
    triton.Config({
            'BLOCK_M': BM,
            'BLOCK_N': BN,
            'NUM_CONSUMER_GROUPS': 1,
            },
            num_stages=s,
            num_warps=w,
            num_consumer_groups=0,
            num_buffers_warp_spec=0,
        ) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4, 6]\
    for w in [4, 8]\
]

#configs_tma = [
#    triton.Config({
#            'BLOCK_M': BM,
#            'BLOCK_N': BN,
#            'NUM_CONSUMER_GROUPS': 1,
#            },
#            num_stages=s,
#            num_warps=w,
#            num_consumer_groups=1,
#            num_buffers_warp_spec=1,
#        ) \
#    for BM in [128]\
#    for BN in [64, 128]\
#    for s in [1, 2, 3, 4, 6]\
#    for w in [4]\
#]

configs_tma = [
    triton.Config({
            'BLOCK_M': BM,
            'BLOCK_N': BN,
            'NUM_CONSUMER_GROUPS': 2,
            },
            num_stages=s,
            num_warps=w,
            num_consumer_groups=2,
            num_buffers_warp_spec=1,
        ) \
    for BM in [128]\
    for BN in [128]\
    for s in [2]\
    for w in [4]\
]


def keep_tma(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if (torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8):
        return False
    return True


filename = 'dump/bn64/28/_attn_fwd_tma.ptx'
filename = 'dump/bn64/110/_attn_fwd_tma.ptx'
filename = 'dump/bn/116/_attn_fwd_tma.ptx'
#filename = 'dump/bn128/IXAJR2I3YEALASAE63LPXXULMZ7LXYJCDW63W6Z672AFOTPPXJMA/_attn_fwd_tma.ptx'
filename = 'dump/bn128/consumer2buffer1stage2/_attn_fwd_tma.ttgir'

#@txl.autotune(
@triton.autotune(
        configs=list(filter(keep_tma, configs_tma)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
        use_cuda_graph=True
        )
#@txl.jit(diff_mode='ttir', log_dir='dump')
#@txl.jit(diff_mode='ttgir')
@txl.jit
#@txl.jit(src_file=filename)
def _attn_fwd_tma(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  NUM_CONSUMER_GROUPS,
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr,  #
                  Q_layout,
                  ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    #tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    #offset_y = off_z + off_h * N_CTX
    offset_y = off_hz * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    #acc1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q = tl._experimental_descriptor_load(desc_q, [qo_offset_y, 0], [BLOCK_M, HEAD_DIM], dtype)

    #q_tma = txl.subtma(desc_q, [off_z, off_h, start_m * BLOCK_M], Q_layout, [BLOCK_M, HEAD_DIM], dtype)
    #q = txl.load(q_tma)

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX,  #
                                            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX,  #
                                            )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m

    tl.store(m_ptrs, m_i)
    tl._experimental_descriptor_store(desc_o, acc.to(dtype), [qo_offset_y, 0])
    #tl._experimental_descriptor_store(desc_o, acc.to(dtype), [0, 0])


@txl.jit
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
@txl.jit
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
@txl.jit
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


@txl.jit
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


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, USE_TMA=True, no_tune=False):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        Q_layout = txl.Layout(
            q.shape,
            order=txl.OrderType.RIGHT,
            order_map=(1, 0)
        )
        K_layout = txl.Layout(
            k.shape,
            (None, None),
            order=txl.OrderType.RIGHT,
            order_map=txl.OrderType.LEFT
        )
        V_layout = txl.Layout(
            v.shape,
            (None, None),
            order=txl.OrderType.RIGHT,
            order_map=txl.OrderType.RIGHT
        )
        O_layout = txl.Layout(
            o.shape,
            (None, None),
            order=txl.OrderType.RIGHT,
            order_map=txl.OrderType.RIGHT
        )
        M_layout = txl.Layout(
            M.shape,
            (None, ),
            order=txl.OrderType.RIGHT,
            order_map=txl.OrderType.RIGHT
        )
        if USE_TMA and supports_tma() and not (torch.cuda.get_device_capability()[0] == 9
                                               and q.dtype == torch.float8_e5m2):
            if no_tune:
                # BLOCK_M: 64, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 2
                #extra_kern_args.update({'BLOCK_M':64, 'BLOCK_N':64, 'num_stages': 2, 'num_warps':4})
                extra_kern_args.update({'BLOCK_M':64, 'BLOCK_N':32, 'num_stages': 2, 'num_warps':4, 'num_consumer_groups': 1, 'num_buffers_warp_spec': 1,})

            K_layout = txl.Layout(
                k.shape,
                (None, None),
                order=txl.OrderType.RIGHT,
                order_map=txl.OrderType.RIGHT
            )
            if True:
                # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
                y_dim = q.shape[0] * q.shape[1] * q.shape[2]

                desc_helper = TmaAutoTuneHelper()
                desc_helper.init_tma_descriptor("q")
                desc_helper.init_tma_descriptor("v")
                desc_helper.init_tma_descriptor("k")
                desc_helper.init_tma_descriptor("o")

                def grid(META):
                    nonlocal desc_helper

                    desc_helper.fill_2d_tma_descriptor("q", q.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"]//META["NUM_CONSUMER_GROUPS"], HEAD_DIM_K,
                                                       q.element_size())

                    desc_helper.fill_2d_tma_descriptor("v", v.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                                       v.element_size())

                    desc_helper.fill_2d_tma_descriptor("k", k.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                                       k.element_size())

                    desc_helper.fill_2d_tma_descriptor("o", o.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"]//META["NUM_CONSUMER_GROUPS"], HEAD_DIM_K,
                                                       o.element_size())

                    return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

                desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
                desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
                desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
                desc_o = desc_helper.get_tma_descriptor_kernel_param("o")

            else:
                desc_q, hook_q = txl.TmaDesc(q, Q_layout)
                desc_k, hook_k = txl.TmaDesc(k, K_layout)
                desc_v, hook_v = txl.TmaDesc(v, V_layout)
                desc_o, hook_o = txl.TmaDesc(o, O_layout)
                def _cpu_pre_hook(nargs):
                    hook_q((nargs["BLOCK_M"]//2, HEAD_DIM_K))
                    hook_k((nargs["BLOCK_N"], HEAD_DIM_K))
                    hook_v((nargs["BLOCK_N"], HEAD_DIM_K))
                    hook_o((nargs["BLOCK_M"]//2, HEAD_DIM_K))
                    return
                if no_tune:
                    _cpu_pre_hook(extra_kern_args)
                else:
                    extra_kern_args["_cpu_pre_hook"] = _cpu_pre_hook
                def grid(META):
                    return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)



            ctx.grid = grid
            _attn_fwd_tma[grid](
                sm_scale, M,  #
                q.shape[0], q.shape[1],  #
                desc_q, desc_k, desc_v, desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                Q_layout=Q_layout,
                **extra_kern_args)
        else:

            #Q_layout = txl.Layout(
            #    q,
            #    order_map=(1, 0)
            #)
            #K_layout = txl.Layout(
            #    k,
            #    (None, None),
            #    order_map=txl.OrderType.LEFT
            #)
            #V_layout = txl.Layout(
            #    v,
            #    (None, None),
            #    order_map=txl.OrderType.RIGHT
            #)
            #O_layout = txl.Layout(
            #    o,
            #    (None, None),
            #    order_map=txl.OrderType.RIGHT
            #)
            #M_layout = txl.Layout(
            #    M,
            #    (None, ),
            #    order_map=txl.OrderType.RIGHT
            #)
            if no_tune:
                extra_kern_args.update({'BLOCK_M':64, 'BLOCK_N':32, 'num_stages': 1, 'num_warps':4})
            grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
            ctx.grid = grid
            _attn_fwd[grid](
                q, k, v, sm_scale, M, o,  # BHSD, M - BHS
                Q_layout, K_layout, V_layout, O_layout, M_layout,
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  # 3 for causal
                **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
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

        return dq, dk, dv, None, None


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16, no_tune=False):
    #torch.manual_seed(20)
    #q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    #k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    #v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    q = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
    k = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
    v = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
    sm_scale = 0.5
    tri_out = attention(q, k, v, causal, sm_scale, HAS_TMA_DESC, no_tune).half()
    print(tri_out)

    REF = True
    if REF:
        dout = torch.randn_like(q)
        # reference implementation
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale # this causes the result of tri attention to non-0s, in TMA mode
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        # p = torch.exp(p)
        ref_out = torch.matmul(p, v)
    #ref_out.backward(dout)
    #ref_dv, v.grad = v.grad.clone(), None
    #ref_dk, k.grad = k.grad.clone(), None
    #ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    #tri_out = attention(q, k, v, causal, sm_scale, HAS_TMA_DESC, True).half()
    #tri_out.backward(dout)
    #tri_dv, v.grad = v.grad.clone(), None
    #tri_dk, k.grad = k.grad.clone(), None
    #tri_dq, q.grad = q.grad.clone(), None
    # compare
    #NOTE:
    # TMA will produce 0s if not having other kernels executed (like torch.matmul)
    # non-TMA is more accurate than TMA version, but under 4x32x1024x64, both are not fufilling the allclose
    # this correctness issue only happens when get too many B and H (maybe S also)
    print(torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0.0))
    #rtol = 0.0
    ## Relative tolerance workaround for known hardware limitation of MI200 GPU.
    ## For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    #if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
    #    rtol = 1e-2
    #assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    #assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    #assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))

configs0 = []
for mode in ["fwd"]:
    for causal in [True]:
        if mode == "bwd" and not causal:
            continue
        configs0.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 11)],
                line_arg="provider",
                line_vals=["triton-fp16"],
                line_names=["Triton [FP16]"],
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs0)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE, no_tune=False):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, HAS_TMA_DESC, no_tune)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=1000, rep=1000)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
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


# only works on post-Ampere GPUs right now
if __name__ == "__main__":
    #_attn_fwd_tma
    #_attn_fwd
    no_tune=False
    print("TEST...")
    #test_op(1, 2, 1024, 64, True, dtype=torch.float16, no_tune=no_tune)

    print("BENCH...")
    bench_flash_attention.run(save_path=".", print_data=True, no_tune=no_tune)
