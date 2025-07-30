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

try:
    import txl
    Has_TXL = True
    print("TXL")
except:
    class txl:
        @staticmethod
        def jit(use_txl=False, diff_mode='ttir'):
            def decorator(func):
                return func
            return decorator
    Has_TXL = False
    print("No txl")

# ENABLE_LHS_TO_TMEM is an experimental environment variable for Blackwell.
# If it is set to 1 it can improve performance of Blackwell attention. However,
# it defaults to 0 as it is known to cause correctness issues outside of the
# _attn_fwd_tma kernel below.

#DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device(type='cuda', index=0)


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


###################################################
# Triton + No TMA
###################################################

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
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
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
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
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


###################################################
# Triton + TMA
###################################################

#@triton.jit
@txl.jit(use_txl=False)
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

#no_tune
#@triton.autotune(configs=list(filter(keep_tma, configs_tma)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
#@triton.jit
@txl.jit(use_txl=False)
def _attn_fwd_tma(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr,  #
                  NUM_STAGES: tl.constexpr  #
                  ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z + off_h * N_CTX
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
    q = tl._experimental_descriptor_load(desc_q, [qo_offset_y, 0], [BLOCK_M, HEAD_DIM], dtype)
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

###################################################
# TXL + TMA + multi-stage
###################################################

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs_tma = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4, 6]\
    for w in [4, 8]\
]


def keep_tma(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if (torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8):
        return False
    return True


#no_tune
#@triton.autotune(configs=list(filter(keep_tma, configs_tma)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
#@triton.jit
@txl.jit
def _attn_fwd_tma_txl(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr,  #
                  NUM_STAGES: tl.constexpr,  #
                  NUM_CONSUMERS: tl.constexpr  #
                  ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_hz * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    # These are in regs
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout

    bQ = txl.smem_alloc([BLOCK_M, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ = txl.mbar_alloc(1) # 1 buffer

    bQ0 = txl.get_buffer(bQ, 0)
    pMbar_bQ0 = txl.get_buffer(pMbar_bQ, 0)
    txl.mbar_expect(pMbar_bQ0, BLOCK_M * HEAD_DIM * 2)
    txl.tma_load(bQ0, desc_q, [qo_offset_y, 0], pMbar_bQ0)
    txl.mbar_wait(pMbar_bQ0, 0)
    # can inval pMbar_bQ

    # TODO: func type mismatch

    ## stage 1: off-band
    ## For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    ## For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    #if STAGE & 1:
    #    acc, l_i, m_i = _attn_fwd_inner_tma_txl(acc, l_i, m_i,
    #                                        bQ, pMbar_bQ,
    #                                        desc_k, desc_v,  #
    #                                        offset_y, dtype, start_m, qk_scale,  #
    #                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
    #                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
    #                                        NUM_STAGES)
    ## stage 2: on-band
    ## For causal = True, STAGE = 2, causal last block
    #if STAGE & 2:
    #    # barrier makes it easier for compielr to schedule the
    #    # two loops independently
    #    acc, l_i, m_i = _attn_fwd_inner_tma_txl(acc, l_i, m_i, bQ, pMbar_bQ,  #
    #                                        desc_k, desc_v,  #
    #                                        offset_y, dtype, start_m, qk_scale,  #
    #                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
    #                                        2, offs_m, offs_n, N_CTX,  #
    #                                        NUM_STAGES)
    # range of values handled by this stage
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
    tl._experimental_descriptor_store(desc_o, acc.to(dtype), [qo_offset_y, 0]) # TODO: tma_store


###################################################
# TXL + TMA + FA3 Algo1
###################################################

@txl.jit
#@txl.jit(diff_mode='llir', log_dir='dump')
def _attn_fwd_ws_tma_txl1(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr,  #
                  NUM_STAGES: tl.constexpr,  #
                  NUM_CONSUMERS: tl.constexpr,  #
                  ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_hz * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # initialize offsets
    #offs_n = tl.arange(0, BLOCK_N)

    # load q: it will stay in SRAM throughout

    # init for pipelines:
    # smem
    # producer mbar for each smem tma
    # consumer mbar for each consumer each dot
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
            tl._experimental_descriptor_store(desc_o, acc.to(dtype), [qo_offset_y, 0]) # TODO: tma_store
        if txl.is_warpgroup([2]):
            tl._experimental_descriptor_store(desc_o, acc.to(dtype), [qo_offset_y + BLOCK_M // 2, 0]) # TODO: tma_store

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

        # 4 1024 32 64 : 178 TFLOPS, 156 for pip install
        # 4 1024 32 128 : 227 TFLOPS
        # txl: CAREFUL!
        tma_best_config = {'BLOCK_M':64, 'BLOCK_N':64, 'num_stages': 3, 'num_warps':4, 'NUM_STAGES': 3, 'NUM_CONSUMERS': 1}
        # for ws
        #tma_best_config = {'BLOCK_M':128, 'BLOCK_N':128, 'NUM_CONSUMERS': 2, 'num_stages': 2, 'num_warps':4, 'NUM_STAGES': 2, 'num_warpgroups': 3}
        load_best_config = {'BLOCK_M':64, 'BLOCK_N':32, 'num_stages': 1, 'num_warps':4}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        if USE_TMA and supports_tma() and not (torch.cuda.get_device_capability()[0] == 9
                                               and q.dtype == torch.float8_e5m2):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            if no_tune:
                extra_kern_args.update(tma_best_config)

            desc_helper = TmaAutoTuneHelper()
            desc_helper.init_tma_descriptor("q")
            desc_helper.init_tma_descriptor("v")
            desc_helper.init_tma_descriptor("k")
            desc_helper.init_tma_descriptor("o")

            def grid(META):
                nonlocal desc_helper

                desc_helper.fill_2d_tma_descriptor("q", q.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"] // META["NUM_CONSUMERS"], HEAD_DIM_K,
                                                   q.element_size())

                desc_helper.fill_2d_tma_descriptor("v", v.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                                   v.element_size())

                desc_helper.fill_2d_tma_descriptor("k", k.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                                   k.element_size())

                desc_helper.fill_2d_tma_descriptor("o", o.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"] // META["NUM_CONSUMERS"], HEAD_DIM_K,
                                                   o.element_size())

                return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

            desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
            desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
            desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
            desc_o = desc_helper.get_tma_descriptor_kernel_param("o")

            ctx.grid = grid
            #_attn_fwd_tma[grid](
            _attn_fwd_tma_txl[grid](
            #_attn_fwd_ws_tma_txl1[grid](
                sm_scale, M,  #
                q.shape[0], q.shape[1],  #
                desc_q, desc_k, desc_v, desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage, #
                **extra_kern_args)
        else:
            if no_tune:
                extra_kern_args.update(load_best_config)
            grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
            ctx.grid = grid
            _attn_fwd[grid](
                q, k, v, sm_scale, M, o,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                **extra_kern_args)

        #ctx.save_for_backward(q, k, v, o, M)
        #ctx.sm_scale = sm_scale
        #ctx.HEAD_DIM = HEAD_DIM_K
        #ctx.causal = causal
        return o

attention = _attention.apply

###################################################
### Helpers
###################################################

# Validate results
import sys
sys.path.insert(0, './tools')  # Insert at beginning
from test_util import attention_ref
import math

#@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
#@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16, no_tune=False):
    #torch.manual_seed(20)
    #q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    #k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    #v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    q = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
    k = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
    v = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
    #sm_scale = 0.5
    #sm_scale = 1.0
    q1 = q.permute(0,2,1,3).contiguous()
    k1 = k.permute(0,2,1,3).contiguous()
    v1 = v.permute(0,2,1,3).contiguous()

    # txl
    if Has_TXL:
        tri_out = attention(q, k, v, causal, 1/math.sqrt(HEAD_DIM), HAS_TMA_DESC, no_tune).half()
    elif HAS_FLASH:
        #tri_out = flash_attn_func(q1, k1, v1, softmax_scale=sm_scale, causal=causal).half()
        tri_out = flash_attn_func(q1, k1, v1, causal=causal).half()
        tri_out = tri_out.permute(0,2,1,3).contiguous()

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
    print(f"Output max diff: {(tri_out - ref_out).abs().max().item()}")
    print(f"Output mean diff: {(tri_out - ref_out).abs().mean().item()}")
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

    #rtol = 0.0
    ## Relative tolerance workaround for known hardware limitation of MI200 GPU.
    ## For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    #if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
    #    rtol = 1e-2


try:
    import sys
    sys.path.insert(0, '/ssd2/zhangzn/flatn/hopper')  # Insert at beginning
    from flash_attn_interface import flash_attn_func
    #from flash_attn.flash_attn_interface import \
    #    flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
    print("Has Flash")
except BaseException:
    HAS_FLASH = False


###################################################
### Original config
###################################################

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 16, 32, 128
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
                (["Flash-3"] if HAS_FLASH else []),
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



###################################################
### Forward centric config
###################################################

line_vals = []
line_names = []
if Has_TXL:
    line_vals.append("triton-fp16")
    line_names.append("Triton [FP16]")
if HAS_FLASH:
    line_vals.append("flash")
    line_names.append("Flash-3")
configs0 = []
for mode in ["fwd"]:
    for causal in [False]:
        if mode == "bwd" and not causal:
            continue
        configs0.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 14)],
                line_arg="provider",
                #txl
                line_vals=line_vals,
                line_names=line_vals,
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-tok16k-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
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
    # follow fa3 paper
    BATCH = int(16483 / N_CTX)
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    if "triton" in provider:
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1/math.sqrt(HEAD_DIM)
        fn = lambda: attention(q, k, v, causal, sm_scale, HAS_TMA_DESC, no_tune)
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


# only works on post-Ampere GPUs right now
if __name__ == "__main__":

    no_tune=True # has best config
    #no_tune=False # no best config

    print("TEST...")
    #test_op(1, 2, 1024, 64, False, dtype=torch.float16, no_tune=no_tune)
    test_op(16, 32, 1024, 128, False, dtype=torch.float16, no_tune=no_tune)

    print("BENCH...")
    bench_flash_attention.run(save_path=".", print_data=True, no_tune=no_tune)
