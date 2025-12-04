import torch
import os
import math
import triton
import triton.language as tl

import txl
from triton.tools.tensor_descriptor import TensorDescriptor
import triton.profiler.language as pl
import triton.profiler as proton
from txl.language.semantic import TXLSemantic

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

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

# GPT2 at HEAD_DIM = 64
tma_ws_best_config = {'BLOCK_M':128, 'BLOCK_N':64, 'NUM_CONSUMERS': 2, 'NUM_STAGES': 2}

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
def _attn_fwd_ws_tma_txl3_causal(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #

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
    # If no host desc, then make device desc
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
    lo, hi = 0, (start_m + 1) * BLOCK_M
    mask_begin = start_m * BLOCK_M
    offsetkv_y = offset_y + lo


    if txl.is_warpgroup([0]):
        txl.reg_dealloc(24)

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
        txl.reg_alloc(240)

        if txl.is_warpgroup([1]):
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)

            # first let wg1 to start
            #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
            txl.bar_arrive(8, 256)
        else:
            offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

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
            if FP8_OUTPUT:
                acc = tl.dot(p, cur_bV.T, acc)
            else:
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
            if start_n >= mask_begin:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
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

@torch.no_grad()
def flash_attention_v2(q, k, v, dropout_prob=0.0):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    extra_kern_args = {}
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    dummy_block = [1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    sm_scale = 1 / math.sqrt(HEAD_DIM_K)

    _attn_fwd_ws_tma_txl3_causal[grid](
        sm_scale, M,  #
        q.shape[0], q.shape[1],  #
        desc_q, desc_k, desc_v, desc_o,  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
        **extra_kern_args)

    return o