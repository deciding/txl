import triton
import triton.language as tl
import txl
import torch
import os
import sys
import math
import pytest
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _maybe_desc(x, shape, strides, block_shape):
    if isinstance(x, tl.tensor_descriptor):
        return x
    else:
        return tl.make_tensor_descriptor(x, shape, strides, block_shape)
    
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

ws_cfg = dict(BLOCK_M=64, BLOCK_N=128, NUM_STAGES=2)
def _pre_hook_divn(nargs):
    BM = nargs["BLOCK_M"]
    BN = nargs["BLOCK_N"]
    R0 = nargs["R0"]
    PE = nargs["PE_DIM"]
    if not isinstance(nargs["desc_qhat"], TensorDescriptor):
        return
    nargs["desc_qhat"].block_shape = [BM, R0]
    nargs["desc_qpe"].block_shape = [BM, PE]
    nargs["desc_zkv"].block_shape = [BN, R0]
    nargs["desc_kpe"].block_shape = [BN, PE]
    nargs["desc_o_lat"].block_shape = [BM, R0]
@txl.autotune(
    configs=[txl.Config(ws_cfg, num_stages=2, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook_divn)],
    key=["KV_SEQ_LEN", "R0", "PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_wsNsplit_txl(
    sm_scale, M,          
    Z, H, KV_HEADS,                
    desc_qhat,                     
    desc_zkv,                       
    desc_o_lat,                         
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,              
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= R0)
    tl.static_assert(BLOCK_N % 2 == 0)

    dtype = tl.float16
    start_m = tl.program_id(0)     
    off_hz  = tl.program_id(1)     
    off_z   = off_hz // H
    off_h   = off_hz %  H

    rows_q  = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_qp = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + start_m * BLOCK_M

    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    N2 = BLOCK_N // 2

    bQ  = txl.smem_alloc([BLOCK_M, R0], dtype=dtype)
    mQ  = txl.mbar_alloc(1)
    bPE = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mPE = txl.mbar_alloc(1)

    bZ = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    mbZ = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bKpe = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mbK = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cQK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cQK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cPV = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    qkR_s = txl.smem_alloc([BLOCK_M, BLOCK_N//2], dtype=tl.float32, num_stages=NUM_STAGES)
    rowMaxR_s = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES)
    m_qkR_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)   # WG2 arrive, WG1 wait
    m_rMaxR_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)   # WG2 arrive, WG1 wait
    cpR =  txl.mbar_alloc(128, num_stages=NUM_STAGES)  # WG1 arrive，WG2 wait

    if txl.is_warpgroup([0]):

        bQi = txl.get_buffer(bQ, 0)
        mQi = txl.get_buffer(mQ, 0)
        bPEi = txl.get_buffer(bPE, 0)
        mPEi = txl.get_buffer(mPE, 0)

        txl.mbar_expect(mQi, BLOCK_M * R0 * 2)
        txl.tma_load(bQi, desc_qhat, [qo_off, 0], mQi)
        txl.mbar_wait(mQi, 0)
        txl.mbar_expect(mPEi, BLOCK_M * PE_DIM * 2)
        txl.tma_load(bPEi, desc_qp, [qo_off, 0], mPEi)
        txl.mbar_wait(mPEi, 0)

        bufIdxW, phase = 0, 1
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mbZ = txl.get_buffer(mbZ, bufIdxW)
            cur_bZ = txl.get_buffer(bZ, bufIdxW)
            cur_mbK = txl.get_buffer(mbK, bufIdxW)
            cur_bKpe = txl.get_buffer(bKpe, bufIdxW)

            txl.mbar_wait(txl.get_buffer(cQK1, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(cQK2, bufIdxW), phase)
            txl.mbar_expect(cur_mbK, BLOCK_N * PE_DIM * 2)
            txl.tma_load(cur_bKpe, desc_kpe, [kv_off, 0], cur_mbK)

            txl.mbar_wait(txl.get_buffer(cPV, bufIdxW), phase)
            txl.mbar_expect(cur_mbZ, BLOCK_N * R0 * 2)
            txl.tma_load(cur_bZ, desc_zkv, [kv_off, 0], cur_mbZ)

            kv_off += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase ^= 1

    if txl.is_warpgroup([1]):
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        acc = tl.zeros([BLOCK_M, R0], dtype=tl.float32)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mPE,0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bPEi = txl.get_buffer(bPE,0)

        bufIdxR, phase = 0, 0
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):

            cur_mbZ = txl.get_buffer(mbZ, bufIdxR)
            cur_bZ = txl.get_buffer(bZ, bufIdxR)      
            cur_mbK = txl.get_buffer(mbK, bufIdxR)
            cur_bKpe = txl.get_buffer(bKpe, bufIdxR)  

            txl.mbar_wait(cur_mbK, phase)
            KPU = txl.smem_slice(cur_bKpe, 0, BLOCK_N//2, 0)

            qkL = tl.dot(bPEi, KPU.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(cQK1, bufIdxR))

            txl.mbar_wait(cur_mbZ, phase)
            ZU = txl.smem_slice(cur_bZ, 0, BLOCK_N//2, 0)
            ZD = txl.smem_slice(cur_bZ, BLOCK_N//2, BLOCK_N//2, 0)
            qkL += tl.dot(bQi, ZU.T)
            txl.dot_wait(0)

            # txl.print("qkL Value use smem_slice:")
            txl.print(qkL)

            rMaxL = tl.max(qkL, 1) * qk_scale

            txl.mbar_wait(txl.get_buffer(m_rMaxR_ready, bufIdxR), phase)
            txl.mbar_wait(txl.get_buffer(m_qkR_ready, bufIdxR), phase)
            rMaxR = txl.get_buffer(rowMaxR_s, bufIdxR)
            qkR = txl.get_buffer(qkR_s, bufIdxR)

            rMax = tl.maximum(rMaxL, rMaxR)
            m_ij  = tl.maximum(m_i, rMax)

            pL = tl.math.exp2(qkL * qk_scale - m_ij[:, None])
            pR = tl.math.exp2(qkR * qk_scale - m_ij[:, None])
            txl.mbar_arrive(txl.get_buffer(cpR, bufIdxR))

            l_ij = tl.sum(pL, 1) + tl.sum(pR, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij
            acc   = acc * alpha[:, None]
            m_i   = m_ij

            acc = tl.dot(pL.to(dtype), ZU, acc)
            acc = tl.dot(pR.to(dtype), ZD, acc)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(cPV, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        m_i += tl.math.log2(l_i)
        acc  = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i)
        desc_o_lat.store([qo_off, 0], acc.to(dtype))

    if txl.is_warpgroup([2]):
        qk_scale = sm_scale * 1.44269504
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mPE,0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bPEi = txl.get_buffer(bPE,0)

        bufIdxR, phase = 0, 0
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):

            cur_mbZ = txl.get_buffer(mbZ, bufIdxR)
            cur_bZ = txl.get_buffer(bZ, bufIdxR)      
            cur_mbK = txl.get_buffer(mbK, bufIdxR)
            cur_bKpe = txl.get_buffer(bKpe, bufIdxR)  

            txl.mbar_wait(cur_mbK, phase)
            KPD = txl.smem_slice(cur_bKpe, BLOCK_N//2, BLOCK_N//2, 0)

            qkR = tl.dot(bPEi, KPD.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(cQK2, bufIdxR))

            txl.mbar_wait(cur_mbZ, phase)
            ZD = txl.smem_slice(cur_bZ, BLOCK_N//2, BLOCK_N//2, 0)
            qkR += tl.dot(bQi, ZD.T)
            txl.dot_wait(0)

            rMaxR = tl.max(qkR, 1) * qk_scale
            txl.mbar_wait(txl.get_buffer(cpR, bufIdxR), phase^1)

            txl.smem_store(txl.get_buffer(rowMaxR_s, bufIdxR), rMaxR)
            txl.mbar_arrive(txl.get_buffer(m_rMaxR_ready, bufIdxR))

            txl.smem_store(txl.get_buffer(qkR_s, bufIdxR), qkR)
            txl.mbar_arrive(txl.get_buffer(m_qkR_ready, bufIdxR))
            
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

ws_cfg = dict(BLOCK_M=64, BLOCK_N=128, NUM_STAGES=2)
def _pre_hook_divn(nargs):
    BM = nargs["BLOCK_M"]
    BN = nargs["BLOCK_N"] // 2
    R0 = nargs["R0"]
    PE = nargs["PE_DIM"]
    if not isinstance(nargs["desc_qhat"], TensorDescriptor):
        return
    nargs["desc_qhat"].block_shape = [BM, R0]
    nargs["desc_qpe"].block_shape = [BM, PE]
    nargs["desc_zkv"].block_shape = [BN, R0]
    nargs["desc_kpe"].block_shape = [BN, PE]
    nargs["desc_o_lat"].block_shape = [BM, R0]
@txl.autotune(
    configs=[txl.Config(ws_cfg, num_stages=2, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook_divn)],
    key=["KV_SEQ_LEN", "R0", "PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_wsNsplit_txl_debug(
    sm_scale, M,          
    Z, H, KV_HEADS,                
    desc_qhat,                     
    desc_zkv,                       
    desc_o_lat,                         
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,              
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= R0)
    tl.static_assert(BLOCK_N % 2 == 0)

    dtype = tl.float16
    start_m = tl.program_id(0)     
    off_hz  = tl.program_id(1)     
    off_z   = off_hz // H
    off_h   = off_hz %  H

    rows_q  = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_qp = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + start_m * BLOCK_M

    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    N2 = BLOCK_N // 2

    bQ  = txl.smem_alloc([BLOCK_M, R0], dtype=dtype)
    mQ  = txl.mbar_alloc(1)
    bPE = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mPE = txl.mbar_alloc(1)

    bZU = txl.smem_alloc([BLOCK_N// 2, R0], dtype=dtype, num_stages=NUM_STAGES)
    mbZU = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bZD = txl.smem_alloc([BLOCK_N// 2, R0], dtype=dtype, num_stages=NUM_STAGES)
    mbZD = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bKpeU = txl.smem_alloc([BLOCK_N// 2, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mbKU = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bKpeD = txl.smem_alloc([BLOCK_N// 2, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mbKD = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cQK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cQK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cPV = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    qkR_s = txl.smem_alloc([BLOCK_M, BLOCK_N//2], dtype=tl.float32, num_stages=NUM_STAGES)
    rowMaxR_s = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES)
    m_qkR_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)   # WG2 arrive, WG1 wait
    m_rMaxR_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)   # WG2 arrive, WG1 wait
    cpR =  txl.mbar_alloc(128, num_stages=NUM_STAGES)  # WG1 arrive，WG2 wait

    if txl.is_warpgroup([0]):

        bQi = txl.get_buffer(bQ, 0)
        mQi = txl.get_buffer(mQ, 0)
        bPEi = txl.get_buffer(bPE, 0)
        mPEi = txl.get_buffer(mPE, 0)

        txl.mbar_expect(mQi, BLOCK_M * R0 * 2)
        txl.tma_load(bQi, desc_qhat, [qo_off, 0], mQi)
        txl.mbar_wait(mQi, 0)
        txl.mbar_expect(mPEi, BLOCK_M * PE_DIM * 2)
        txl.tma_load(bPEi, desc_qp, [qo_off, 0], mPEi)
        txl.mbar_wait(mPEi, 0)

        bufIdxW, phase = 0, 1
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mbZU = txl.get_buffer(mbZU, bufIdxW)
            cur_bZU = txl.get_buffer(bZU, bufIdxW)
            cur_mbZD = txl.get_buffer(mbZD, bufIdxW)
            cur_bZD = txl.get_buffer(bZD, bufIdxW)
            cur_mbKU = txl.get_buffer(mbKU, bufIdxW)
            cur_bKpeU = txl.get_buffer(bKpeU, bufIdxW)
            cur_mbKD = txl.get_buffer(mbKD, bufIdxW)
            cur_bKpeD = txl.get_buffer(bKpeD, bufIdxW)

            txl.mbar_wait(txl.get_buffer(cQK1, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(cQK2, bufIdxW), phase)
            txl.mbar_expect(cur_mbKU, BLOCK_N //2 * PE_DIM * 2)
            txl.tma_load(cur_bKpeU, desc_kpe, [kv_off, 0], cur_mbKU)
            txl.mbar_expect(cur_mbKD, BLOCK_N //2 * PE_DIM * 2)
            txl.tma_load(cur_bKpeD, desc_kpe, [kv_off + BLOCK_N//2, 0], cur_mbKD)

            txl.mbar_wait(txl.get_buffer(cPV, bufIdxW), phase)
            txl.mbar_expect(cur_mbZU, BLOCK_N // 2 * R0 * 2)
            txl.tma_load(cur_bZU, desc_zkv, [kv_off, 0], cur_mbZU)
            txl.mbar_expect(cur_mbZD, BLOCK_N // 2 * R0 * 2)
            txl.tma_load(cur_bZD, desc_zkv, [kv_off + BLOCK_N//2, 0], cur_mbZD)

            kv_off += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase ^= 1

    if txl.is_warpgroup([1]):
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        acc = tl.zeros([BLOCK_M, R0], dtype=tl.float32)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mPE,0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bPEi = txl.get_buffer(bPE,0)

        bufIdxR, phase = 0, 0
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):

            cur_mbZU = txl.get_buffer(mbZU, bufIdxR)
            cur_bZU = txl.get_buffer(bZU, bufIdxR) 
            cur_mbZD = txl.get_buffer(mbZD, bufIdxR)
            cur_bZD = txl.get_buffer(bZD, bufIdxR)     
            cur_mbKU = txl.get_buffer(mbKU, bufIdxR)
            cur_bKpeU = txl.get_buffer(bKpeU, bufIdxR)  

            txl.mbar_wait(cur_mbKU, phase)

            qkL = tl.dot(bPEi, cur_bKpeU.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(cQK1, bufIdxR))

            txl.mbar_wait(cur_mbZU, phase)
            qkL += tl.dot(bQi, cur_bZU.T)
            txl.dot_wait(0)

            # txl.print("triton2:")
            # txl.print(qkL)

            rMaxL = tl.max(qkL, 1) * qk_scale

            txl.mbar_wait(txl.get_buffer(m_rMaxR_ready, bufIdxR), phase)
            txl.mbar_wait(txl.get_buffer(m_qkR_ready, bufIdxR), phase)
            rMaxR = txl.get_buffer(rowMaxR_s, bufIdxR)
            qkR = txl.get_buffer(qkR_s, bufIdxR)

            rMax = tl.maximum(rMaxL, rMaxR)
            m_ij  = tl.maximum(m_i, rMax)

            pL = tl.math.exp2(qkL * qk_scale - m_ij[:, None])
            pR = tl.math.exp2(qkR * qk_scale - m_ij[:, None])
            txl.mbar_arrive(txl.get_buffer(cpR, bufIdxR))

            l_ij = tl.sum(pL, 1) + tl.sum(pR, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij
            acc   = acc * alpha[:, None]
            m_i   = m_ij

            acc = tl.dot(pL.to(dtype), cur_bZU, acc)
            txl.mbar_wait(cur_mbZD, phase)
            acc = tl.dot(pR.to(dtype), cur_bZD, acc)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(cPV, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        m_i += tl.math.log2(l_i)
        acc  = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i)
        desc_o_lat.store([qo_off, 0], acc.to(dtype))

    if txl.is_warpgroup([2]):
        qk_scale = sm_scale * 1.44269504
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mPE,0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bPEi = txl.get_buffer(bPE,0)

        bufIdxR, phase = 0, 0
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):

            cur_mbZD = txl.get_buffer(mbZD, bufIdxR)
            cur_bZD = txl.get_buffer(bZD, bufIdxR)      
            cur_mbKD = txl.get_buffer(mbKD, bufIdxR)
            cur_bKpeD = txl.get_buffer(bKpeD, bufIdxR)  

            txl.mbar_wait(cur_mbKD, phase)

            qkR = tl.dot(bPEi, cur_bKpeD.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(cQK2, bufIdxR))

            txl.mbar_wait(cur_mbZD, phase)
            qkR += tl.dot(bQi, cur_bZD.T)
            txl.dot_wait(0)

            rMaxR = tl.max(qkR, 1) * qk_scale
            txl.mbar_wait(txl.get_buffer(cpR, bufIdxR), phase^1)

            txl.smem_store(txl.get_buffer(rowMaxR_s, bufIdxR), rMaxR)
            txl.mbar_arrive(txl.get_buffer(m_rMaxR_ready, bufIdxR))

            txl.smem_store(txl.get_buffer(qkR_s, bufIdxR), qkR)
            txl.mbar_arrive(txl.get_buffer(m_qkR_ready, bufIdxR))
            
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

def mla_test(q, kv, qpe, kpe, sm_scale, algo=0):
    HEAD_DIM_Q = q.shape[-1]
    HEAD_DIM_Z = kv.shape[-1]
    HEAD_DIM_PE = qpe.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_Z
    o = torch.empty_like(q)
    stage = 3
    extra_kern_args = {}

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    if supports_host_descriptor():
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]
        kv_dim = kv.shape[0] * kv.shape[1] * kv.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
        desc_kv = TensorDescriptor(kv, shape=[kv_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
        desc_qpe = TensorDescriptor(qpe, shape=[y_dim, HEAD_DIM_PE], strides=[HEAD_DIM_PE, 1], block_shape=dummy_block)
        desc_kpe = TensorDescriptor(kpe, shape=[kv_dim, HEAD_DIM_PE], strides=[HEAD_DIM_PE, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
    else:
        desc_q = q
        desc_kv = kv
        desc_kpe = kpe
        desc_qpe = qpe
        desc_o = o

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    algo_map = {
        2: mla_decode_latent_sharedZ_wsNsplit_txl,
        5: mla_decode_latent_sharedZ_wsNsplit_txl_debug
    }

    algo_map[algo][grid](
        sm_scale, M,
        q.shape[0], q.shape[1], kv.shape[1],
        desc_q,
        desc_kv,
        desc_o,
        q.shape[2], kv.shape[2],
        desc_qpe, desc_kpe,
        PE_DIM=HEAD_DIM_PE,
        R0=HEAD_DIM_Z,
        **extra_kern_args
    )

    return o

def ref_mla(q, kv, qpe, kpe, sm_scale):
    Z, H, N_Q, R0 = q.shape
    _, KV_HEADS, KV_SEQ_LEN, _ = kv.shape
    PE_DIM = qpe.shape[-1]

    heads_per_kv = H // KV_HEADS
    o = torch.empty_like(q)
    m_all = torch.empty((Z, H, N_Q), device=q.device, dtype=torch.float32)

    for z in range(Z):
        for h in range(H):
            kv_head_idx = h // heads_per_kv
            q_ = q[z, h]  # [N_Q, R0]
            qpe_ = qpe[z, h]  # [N_Q, PE_DIM]
            kv_ = kv[z, kv_head_idx]  # [KV_SEQ_LEN, R0]
            kpe_ = kpe[z, kv_head_idx]  # [KV_SEQ_LEN, PE_DIM]

            qk = (q_ @ kv_.T) + (qpe_ @ kpe_.T)  # [N_Q, KV_SEQ_LEN]
            qk = qk * sm_scale * 1.44269504  # log2(e)=1.44269504
            m_i = torch.max(qk, dim=1).values
            p = torch.pow(2, qk - m_i[:, None])
            l_i = torch.sum(p, dim=1)
            m_all[z, h] = m_i + torch.log2(l_i)
            o[z, h] = (p @ kv_) / l_i[:, None]
    return o

def test_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0, no_tune=False):
    # q = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    # kv = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    # qpe = (torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    # kpe = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    q = (torch.ones((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    # the up region of kv is 0.5, the down region is 2.0
    kvu = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN//2, R0), dtype=dtype, device=DEVICE)*0.5)
    kvd = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN//2, R0), dtype=dtype, device=DEVICE)*2.0)
    kv = torch.cat([kvu, kvd], dim=2)
    qpe = (torch.ones((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    kpe = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    # the result of qkL should be 64*1*1(pe part) + 128*1*0.5=128, the result of qkR should be 64*1*1 + 128*1*2=320
    # however, the result of qkR when using smem_slice is 64 + 128*1*(0.5+2.0)/2 =224

    sm_scale = 1 / math.sqrt(R0)

    tri_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=algo)
    debug_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=5)
    ref_out = ref_mla(q, kv, qpe, kpe, sm_scale)

    max_err = (tri_out - ref_out).abs().max().item()
    print(f"smem_slice out: {tri_out}")
    print(f"tma out: {debug_out}")
    print(f"ref out: {ref_out}")
    print(f"Z{Z} H{H} NQ{N_Q} KH{KV_HEADS} KS{KV_SEQ_LEN} R0{R0} PE{PE_DIM} | max err: {max_err:.6f}")

if __name__ == "__main__":
    no_tune=True

    print("TEST...")
    
    test_op(1, 1, 64, 1, 128, 128, 64, algo=2, no_tune=no_tune)