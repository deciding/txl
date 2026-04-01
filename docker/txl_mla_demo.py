import triton
import triton.language as tl
import txl
import torch
import os
import sys
import math
import pytest
from triton.tools.tensor_descriptor import TensorDescriptor
import triton.profiler as proton
from contextlib import contextmanager

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _maybe_desc(x, shape, strides, block_shape):
    if isinstance(x, tl.tensor_descriptor):
        return x
    else:
        return tl.make_tensor_descriptor(x, shape, strides, block_shape)

def _pre_hook(nargs):
    BM = nargs["BLOCK_M"]
    BN = nargs["BLOCK_N"]
    R0 = nargs["R0"]
    PE = nargs["PE_DIM"]
    # 这里的TensorDescriptor是host端，我们在host端生成descriptor时是没有设置block shape的，所以pre_hook里要设置
    # 如果是device端，可以看到kernel里是有用_maybe_desc函数生成descriptor的
    if not isinstance(nargs["desc_qhat"], TensorDescriptor):
        return
    nargs["desc_qhat"].block_shape = [BM, R0]
    nargs["desc_qpe"].block_shape = [BM, PE]
    nargs["desc_zkv"].block_shape = [BN, R0]
    nargs["desc_kpe"].block_shape = [BN, PE]
    nargs["desc_o_lat"].block_shape = [BM, R0]

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

tma_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)

@txl.autotune(
  configs=[txl.Config(tma_cfg, num_stages=1, num_warps=4, num_warpgroups=1, pre_hook=_pre_hook)],
  key=["KV_SEQ_LEN", "R0", "PE_DIM"] # 如果这三个参数没有变化，就使用同样的配置
)
@txl.jit
def mla_decode_latent_sharedZ_txl(
    sm_scale, M, 
    Z, H, KV_HEADS, 
    desc_qhat, # 输入Q [Z,H,N_Q,r0]，核外预处理原本的Q
    desc_zkv,  # Zkv [Z,KV_HEADS,KV_SEQ_LEN,r0]
    desc_o_lat, # 输出Y [Z,H,N_Q,r0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr, 
):
    tl.static_assert(BLOCK_N <= R0)

    dtype = tl.float16
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)  
    off_z = off_hz // H
    off_h = off_hz %  H

    rows_q  = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_zkv  = _maybe_desc(desc_zkv,  [rows_kv,R0], [R0, 1], [BLOCK_N, R0])
    desc_o_lat = _maybe_desc(desc_o_lat,[rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q,  PE_DIM], [PE_DIM,1], [BLOCK_M, PE_DIM])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM,1], [BLOCK_N, PE_DIM])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + pid_m * BLOCK_M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, R0], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # 载入 Qhat & Q_pe
    bQ = txl.smem_alloc([BLOCK_M, R0], dtype=dtype) 
    mQ = txl.mbar_alloc(1)
    bQ0 = txl.get_buffer(bQ,0); 
    mQ0 =txl.get_buffer(mQ,0)
    txl.mbar_expect(mQ0, BLOCK_M*R0*2); 
    txl.tma_load(bQ0, desc_qhat, [qo_off,0], mQ0)
    txl.mbar_wait(mQ0,0)

    bP = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype) 
    mP = txl.mbar_alloc(1)
    bP0 = txl.get_buffer(bP,0); 
    mP0 = txl.get_buffer(mP,0)
    txl.mbar_expect(mP0, BLOCK_M*PE_DIM*2); 
    txl.tma_load(bP0, desc_qpe, [qo_off,0], mP0)
    txl.mbar_wait(mP0,0)

    # 载入 Zkv & K_pe 
    bZ  = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    mZ  = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bKp = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mKp = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    bufIdxW = 0
    bufIdxR = 0
    phase = 0

    for i in tl.static_range(0, NUM_STAGES):
        cur_mZ = txl.get_buffer(mZ, bufIdxW)
        cur_Z = txl.get_buffer(bZ, bufIdxW)
        txl.mbar_expect(cur_mZ, BLOCK_N*R0*2)
        txl.tma_load(cur_Z, desc_zkv, [kv_off,0], cur_mZ)
        cur_mKp = txl.get_buffer(mKp, bufIdxW)
        cur_Kp = txl.get_buffer(bKp, bufIdxW)
        txl.mbar_expect(cur_mKp, BLOCK_N*PE_DIM*2)
        txl.tma_load(cur_Kp, desc_kpe, [kv_off,0], cur_mKp)
        
        kv_off += BLOCK_N
        bufIdxW = (bufIdxW + 1) % NUM_STAGES

    for start_n in range(0, KV_SEQ_LEN, BLOCK_N):
        cur_mZ = txl.get_buffer(mZ, bufIdxR)
        cur_Z = txl.get_buffer(bZ, bufIdxR)
        txl.mbar_wait(cur_mZ, phase)
        qk = tl.dot(txl.get_buffer(bQ,0), cur_Z.T)
        txl.dot_wait(0)

        cur_mKp = txl.get_buffer(mKp, bufIdxR)
        cur_Kp = txl.get_buffer(bKp, bufIdxR)
        txl.mbar_wait(cur_mKp, phase)
        qk += tl.dot(txl.get_buffer(bP,0), cur_Kp.T)
        txl.dot_wait(0)

        m_ij = tl.maximum(m_i, tl.max(qk,1) * qk_scale)
        qk = qk * qk_scale - m_ij[:,None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:,None]
        m_i = m_ij

        acc = tl.dot(p.to(dtype), cur_Z, acc)
        txl.dot_wait(0)

        bufIdxR = (bufIdxR + 1) % NUM_STAGES
        if bufIdxR == 0:
            phase = phase ^ 1

        if start_n < KV_SEQ_LEN - (NUM_STAGES-1)*BLOCK_N:
            cur_mZ = txl.get_buffer(mZ, bufIdxW)
            cur_Z = txl.get_buffer(bZ, bufIdxW)
            txl.mbar_expect(cur_mZ, BLOCK_N*R0*2)
            txl.tma_load(cur_Z, desc_zkv, [kv_off,0], cur_mZ)
            cur_mKp = txl.get_buffer(mKp, bufIdxW)
            cur_Kp = txl.get_buffer(bKp, bufIdxW)
            txl.mbar_expect(cur_mKp, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_Kp, desc_kpe, [kv_off,0], cur_mKp)
            
            kv_off += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:,None]
    m_ptrs = M + off_hz * N_Q + offs_m
    tl.store(m_ptrs, m_i)
    desc_o_lat.store([qo_off, 0], acc.to(dtype))

def _pre_hook_divm(nargs):
    BM = nargs["BLOCK_M"] // 2
    BN = nargs["BLOCK_N"]
    R0 = nargs["R0"]
    PE = nargs["PE_DIM"]
    # 这里的TensorDescriptor是host端，我们在host端生成descriptor时是没有设置block shape的，所以pre_hook里要设置
    # 如果是device端，可以看到kernel里是有用_maybe_desc函数生成descriptor的
    if not isinstance(nargs["desc_qhat"], TensorDescriptor):
        return
    nargs["desc_qhat"].block_shape = [BM, R0]
    nargs["desc_qpe"].block_shape = [BM, PE]
    nargs["desc_zkv"].block_shape = [BN, R0]
    nargs["desc_kpe"].block_shape = [BN, PE]
    nargs["desc_o_lat"].block_shape = [BM, R0]

# 需要调整
ws_cfg = dict(BLOCK_M=128, BLOCK_N=64, NUM_STAGES=1)

@txl.autotune(
    configs=[txl.Config(ws_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook_divm)],
    key=["KV_SEQ_LEN", "R0", "PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_ws_txl(
    sm_scale, M,
    Z, H, KV_HEADS,
    desc_qhat, # [Z,H,N_Q,R0]
    desc_zkv, # [Z,KV_HEADS,KV_SEQ_LEN,R0]
    desc_o_lat, # [Z,H,N_Q,R0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe, # Q_pe: [Z,H,N_Q,PE_DIM], K_pe: [Z,KV_HEADS,KV_SEQ_LEN,PE_DIM]
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):   
    tl.static_assert(BLOCK_N <= R0)

    dtype = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    rows_q = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])
    # 有误，block shape必须与写回的对应
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + start_m * BLOCK_M

    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    bQ0 = txl.smem_alloc([BLOCK_M//2, R0], dtype=dtype)
    mQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, R0], dtype=dtype)
    mQ1 = txl.mbar_alloc(1)
    bP0 = txl.smem_alloc([BLOCK_M//2, PE_DIM], dtype=dtype)
    mP0 = txl.mbar_alloc(1)
    bP1 = txl.smem_alloc([BLOCK_M//2, PE_DIM], dtype=dtype)
    mP1 = txl.mbar_alloc(1)

    bZ = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    mbZ = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    bKpe = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mbK = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cQK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cPV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cQK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cPV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    if txl.is_warpgroup([0]):
        bQ0i, bQ1i = txl.get_buffer(bQ0,0), txl.get_buffer(bQ1,0)
        bP0i, bP1i = txl.get_buffer(bP0,0), txl.get_buffer(bP1,0)
        mQ0i, mQ1i = txl.get_buffer(mQ0,0), txl.get_buffer(mQ1,0)
        mP0i, mP1i = txl.get_buffer(mP0,0), txl.get_buffer(mP1,0)

        txl.mbar_expect(mQ0i, (BLOCK_M//2)*R0*2)
        txl.tma_load(bQ0i, desc_qhat, [qo_off, 0], mQ0i)
        txl.mbar_wait(mQ0i, 0)
        txl.mbar_expect(mQ1i, (BLOCK_M//2)*R0*2)
        txl.tma_load(bQ1i, desc_qhat, [qo_off + BLOCK_M//2, 0], mQ1i)
        txl.mbar_wait(mQ1i, 0)

        txl.mbar_expect(mP0i, (BLOCK_M//2)*PE_DIM*2)
        txl.tma_load(bP0i, desc_qpe, [qo_off, 0], mP0i)
        txl.mbar_wait(mP0i, 0)
        txl.mbar_expect(mP1i, (BLOCK_M//2)*PE_DIM*2)
        txl.tma_load(bP1i, desc_qpe, [qo_off + BLOCK_M//2, 0], mP1i)
        txl.mbar_wait(mP1i, 0)

        bufIdxW, phase = 0, 1  
        for _start_n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mbZ = txl.get_buffer(mbZ, bufIdxW)
            cur_bZ = txl.get_buffer(bZ, bufIdxW)
            cur_mbK  = txl.get_buffer(mbK,  bufIdxW)
            cur_bKpe = txl.get_buffer(bKpe, bufIdxW)

            txl.mbar_wait(txl.get_buffer(cQK1, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(cQK2, bufIdxW), phase)
            txl.mbar_expect(cur_mbK, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKpe, desc_kpe, [kv_off, 0], cur_mbK)

            txl.mbar_wait(txl.get_buffer(cPV1, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(cPV2, bufIdxW), phase)
            txl.mbar_expect(cur_mbZ, BLOCK_N*R0*2)
            txl.tma_load(cur_bZ, desc_zkv, [kv_off, 0], cur_mbZ)

            kv_off += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase ^= 1

    if txl.is_warpgroup([1]):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)
        qo_off_i = qo_off

        m_i = tl.full([BLOCK_M//2], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M//2], 1.0, tl.float32)
        acc = tl.zeros([BLOCK_M//2, R0], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        txl.mbar_wait(txl.get_buffer(mQ0,0), 0)
        txl.mbar_wait(txl.get_buffer(mP0,0), 0)
        bQ_i  = txl.get_buffer(bQ0,0) 
        bP_i  = txl.get_buffer(bP0,0) 
        cQK_i = cQK1 
        cPV_i = cPV1 

        bufIdxR, phase = 0, 0
        for _start_n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mbK = txl.get_buffer(mbK, bufIdxR)
            cur_bKpe = txl.get_buffer(bKpe, bufIdxR)

            txl.mbar_wait(cur_mbK, phase)
            qk = tl.dot(bP_i, cur_bKpe.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(cQK_i, bufIdxR))

            cur_mbZ = txl.get_buffer(mbZ, bufIdxR)
            cur_bZ = txl.get_buffer(bZ, bufIdxR)
            txl.mbar_wait(cur_mbZ, phase)
            qk += tl.dot(bQ_i, cur_bZ.T)
            txl.dot_wait(0)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            m_i = m_ij

            p = p.to(dtype)
            acc = tl.dot(p, cur_bZ, acc)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(cPV_i, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        m_i += tl.math.log2(l_i)
        acc  = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i)
        # tl.static_print(acc.shape)
        # tl.static_print(desc_o_lat.shape)
        desc_o_lat.store([qo_off_i, 0], acc.to(dtype))
    
    if txl.is_warpgroup([2]):
        offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)
        qo_off_i = qo_off + BLOCK_M//2

        m_i = tl.full([BLOCK_M//2], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M//2], 1.0, tl.float32)
        acc = tl.zeros([BLOCK_M//2, R0], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        txl.mbar_wait(txl.get_buffer(mQ1,0), 0)
        txl.mbar_wait(txl.get_buffer(mP1,0), 0)
        bQ_i  = txl.get_buffer(bQ1,0) 
        bP_i  = txl.get_buffer(bP1,0) 
        cQK_i = cQK2 
        cPV_i = cPV2 

        bufIdxR, phase = 0, 0
        for _start_n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mbK = txl.get_buffer(mbK, bufIdxR)
            cur_bKpe = txl.get_buffer(bKpe, bufIdxR)

            txl.mbar_wait(cur_mbK, phase)
            qk = tl.dot(bP_i, cur_bKpe.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(cQK_i, bufIdxR))

            cur_mbZ = txl.get_buffer(mbZ, bufIdxR)
            cur_bZ = txl.get_buffer(bZ, bufIdxR)
            txl.mbar_wait(cur_mbZ, phase)
            qk += tl.dot(bQ_i, cur_bZ.T)
            txl.dot_wait(0)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            m_i = m_ij

            p = p.to(dtype)
            acc = tl.dot(p, cur_bZ, acc)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(cPV_i, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        m_i += tl.math.log2(l_i)
        acc  = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i)
        desc_o_lat.store([qo_off_i, 0], acc.to(dtype))

def _pre_hook_debug(nargs):
    BM = nargs["BLOCK_M"] // 2
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
    nargs["desc_v"].block_shape = [BN, R0]
ws_cfg = dict(BLOCK_M=128, BLOCK_N=128, NUM_STAGES=2)
@txl.autotune(
    configs=[txl.Config(ws_cfg, num_stages=2, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook_debug)],
    key=["KV_SEQ_LEN", "R0", "PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_ws_debug_txl(
    sm_scale, M,
    Z, H, KV_HEADS,
    desc_qhat, # [Z,H,N_Q,R0]
    desc_zkv, # [Z,KV_HEADS,KV_SEQ_LEN,R0]
    desc_v,
    desc_o_lat, # [Z,H,N_Q,R0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe, # Q_pe: [Z,H,N_Q,PE_DIM], K_pe: [Z,KV_HEADS,KV_SEQ_LEN,PE_DIM]
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):   
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= R0)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_Q

    offset_y = off_z * (N_Q * H) + off_h * N_Q
    qo_offset_y = offset_y + start_m * BLOCK_M


    # load q: it will stay in SRAM throughout
    #q = desc_q.load([qo_offset_y, 0])
    bQ0 = txl.smem_alloc([BLOCK_M//2, R0], dtype=dtype) # bQ has only 1 buffer for reuse only
    pMbar_bQ0 = txl.mbar_alloc(1)
    bQ1 = txl.smem_alloc([BLOCK_M//2, R0], dtype=dtype)
    pMbar_bQ1 = txl.mbar_alloc(1)

    bK = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    bV = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    # TODO: func type mismatch

    # range of values handled by this stage
    lo, hi = 0, KV_SEQ_LEN
    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    offsetkv_y = kv_base


    if txl.is_warpgroup([0]):

        bQ0i = txl.get_buffer(bQ0, 0)
        pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
        bQ1i = txl.get_buffer(bQ1, 0)
        pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

        txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * R0 * 2)
        txl.tma_load(bQ0i, desc_qhat, [qo_offset_y, 0], pMbar_bQ0i)
        txl.mbar_wait(pMbar_bQ0i, 0)
        txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * R0 * 2)
        txl.tma_load(bQ1i, desc_qhat, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
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
            txl.mbar_expect(cur_mbar_bK, BLOCK_N * R0 * 2)
            txl.tma_load(cur_bK, desc_zkv, [offsetkv_y, 0], cur_mbar_bK)

            txl.mbar_wait(cur_mbar_PV1, phase)
            txl.mbar_wait(cur_mbar_PV2, phase)
            txl.mbar_expect(cur_mbar_bV, BLOCK_N * R0 * 2)
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
        acc = tl.zeros([BLOCK_M//2, R0], dtype=tl.float32)
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
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i)

        if txl.is_warpgroup([1]):
            desc_o_lat.store([qo_offset_y, 0], acc.to(dtype))
        if txl.is_warpgroup([2]):
            desc_o_lat.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))



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

            # txl.print("triton 1:")
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

ws_cfg = dict(BLOCK_M=64, BLOCK_N=128, NUM_STAGES=1)
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
    configs=[txl.Config(ws_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook_divn)],
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
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
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
        txl.tma_load(bPEi, desc_qpe, [qo_off, 0], mPEi)
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

        mma_layout: tl.constexpr = txl.NVMMADistributedLayout(
            version=[3, 0],           
            warps_per_cta=[4, 1],     
            instr_shape=[16, 64, 16], 
        )

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
            
            # 不能在这里做qkR_reg *  qk_scale，编译器识别不出qk_scale需要广播成的layout，很奇怪
            qkR_reg = txl.smem_load(qkR, mma_layout)

            rMax = tl.maximum(rMaxL, rMaxR)
            m_ij  = tl.maximum(m_i, rMax)

            pL = tl.math.exp2(qkL * qk_scale - m_ij[:, None])
            pR = tl.math.exp2(qkR_reg - m_ij[:, None])
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
            qkR = qkR * qk_scale

            rMaxR = tl.max(qkR, 1) * qk_scale
            txl.mbar_wait(txl.get_buffer(cpR, bufIdxR), phase^1)

            txl.smem_store(txl.get_buffer(rowMaxR_s, bufIdxR), rMaxR)
            txl.mbar_arrive(txl.get_buffer(m_rMaxR_ready, bufIdxR))

            txl.smem_store(txl.get_buffer(qkR_s, bufIdxR), qkR)
            txl.mbar_arrive(txl.get_buffer(m_qkR_ready, bufIdxR))
            
            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

ws_dim_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)  # 可按需调优

@txl.autotune(
    configs=[txl.Config(ws_dim_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook)],
    key=["KV_SEQ_LEN","R0","PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_ws_dim2_txl( # tilelange ws 
    sm_scale, M,                          
    Z, H, KV_HEADS,
    desc_qhat,                            # [Z,H,N_Q,R0]
    desc_zkv,                             # [Z,KV_HEADS,KV_SEQ_LEN,R0]
    desc_o_lat,                           # [Z,H,N_Q,R0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,                   # Q_pe: [Z,H,N_Q,PE], K_pe: [Z,KV_HEADS,KV_SEQ_LEN,PE]
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= R0)
    tl.static_assert(R0 % 2 == 0)       
    dtype = tl.float16

    pid_m = tl.program_id(0)        
    off_hz = tl.program_id(1)          
    off_z = off_hz // H
    off_h = off_hz %  H

    rows_q = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + pid_m * BLOCK_M
    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    bQ = txl.smem_alloc([BLOCK_M, R0], dtype=dtype)
    bQP  = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mQ = txl.mbar_alloc(1)
    mQP = txl.mbar_alloc(1)

    bZ = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    bKP = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mZ = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mKP = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    mQK = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV_L = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV_R = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    S_shared = txl.smem_alloc([BLOCK_M, BLOCK_N], dtype=dtype, num_stages=NUM_STAGES)
    ALPHA_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES) # 缩放
    SUMEXP_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32) # softmax 分母

    bar_SS_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)  # QK和缩放计算完成
    bar_SS_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)  # QK和缩放使用完毕，可覆写
    bar_sumexp_ready = txl.mbar_alloc(128)  # softmax 分母计算完成

    if txl.is_warpgroup([0]):
        bQi = txl.get_buffer(bQ, 0)
        mQi = txl.get_buffer(mQ, 0)
        bQPi = txl.get_buffer(bQP,  0)
        mQPi = txl.get_buffer(mQP,  0)

        txl.mbar_expect(mQi, BLOCK_M * R0 * 2)
        txl.mbar_expect(mQPi, BLOCK_M * PE_DIM * 2)

        txl.tma_load(bQi, desc_qhat, [qo_off, 0], mQi) 
        txl.mbar_wait(mQi, 0)
        txl.tma_load(bQPi,  desc_qpe, [qo_off, 0], mQPi)
        txl.mbar_wait(mQPi, 0)

        bufIdxW, phase = 0, 1
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mZ = txl.get_buffer(mZ, bufIdxW) 
            cur_bZ = txl.get_buffer(bZ, bufIdxW)
            cur_bKP = txl.get_buffer(bKP, bufIdxW) 
            cur_mKP = txl.get_buffer(mKP, bufIdxW)

            txl.mbar_wait(txl.get_buffer(mQK, bufIdxW), phase)
            txl.mbar_expect(cur_mKP, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP, desc_kpe, [kv_off, 0], cur_mKP)

            txl.mbar_wait(txl.get_buffer(mPV_L, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(mPV_R, bufIdxW), phase)
            txl.mbar_expect(cur_mZ, BLOCK_N*R0*2)
            txl.tma_load(cur_bZ, desc_zkv, [kv_off, 0], cur_mZ)

            kv_off += BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase ^= 1
            
    if txl.is_warpgroup([1]):  # left consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accL= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)
        accL_Output = txl.smem_alloc([BLOCK_M, R0//2], dtype=dtype)
        baccL_Output = txl.get_buffer(accL_Output, 0)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mQP, 0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bQPi = txl.get_buffer(bQP,  0)

        bQli = txl.smem_slice(bQi, 0, R0//2, 1)
        bQri = txl.smem_slice(bQi, R0//2, R0//2, 1)

        bufIdxR, phase = 0, 0
        for _n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mZ = txl.get_buffer(mZ, bufIdxR)
            cur_mKP = txl.get_buffer(mKP, bufIdxR)

            txl.mbar_wait(cur_mKP, phase)
            cur_KP = txl.get_buffer(bKP, bufIdxR)

            acc_s = tl.dot(bQPi, cur_KP.T)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(mQK, bufIdxR))

            txl.mbar_wait(cur_mZ, phase)
            cur_Z = txl.get_buffer(bZ, bufIdxR)
            cur_ZL = txl.smem_slice(cur_Z, 0, R0//2, 1)
            cur_ZR = txl.smem_slice(cur_Z, R0//2, R0//2, 1)

            acc_s += tl.dot(bQli, cur_ZL.T)
            txl.dot_wait(0)
            acc_s += tl.dot(bQri, cur_ZR.T)
            txl.dot_wait(0)

            m_ij  = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
            acc_s = acc_s * qk_scale - m_ij[:, None]
            p     = tl.math.exp2(acc_s)
            l_ij  = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij
            accL  = accL * alpha[:, None]
            m_i   = m_ij

            txl.mbar_wait(txl.get_buffer(bar_SS_free, bufIdxR), phase^1)
            cur_S_shared = txl.get_buffer(S_shared, bufIdxR)
            cur_ALPHA_sh = txl.get_buffer(ALPHA_sh, bufIdxR)
            txl.smem_store(cur_S_shared, p.to(dtype))
            txl.smem_store(cur_ALPHA_sh, alpha)
            txl.mbar_arrive(txl.get_buffer(bar_SS_ready, bufIdxR))

            accL = tl.dot(p.to(dtype), cur_ZL, accL)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(mPV_L, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        txl.smem_store(txl.get_buffer(SUMEXP_sh, 0), l_i)
        txl.mbar_arrive(txl.get_buffer(bar_sumexp_ready, 0))
        m_i += tl.math.log2(l_i)
        accL = accL / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i) 
        # reg -> smem -> gmem
        txl.smem_store(baccL_Output, accL.to(dtype))
        txl.tma_store(baccL_Output, desc_o_lat, [qo_off, 0])            

    if txl.is_warpgroup([2]):  # right consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        accR = tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)
        accR_Output = txl.smem_alloc([BLOCK_M, R0//2], dtype=dtype)
        baccR_Output = txl.get_buffer(accR_Output, 0)

        bufIdxR, phase = 0, 0

        mma_layout: tl.constexpr = txl.NVMMADistributedLayout(
            version=[3, 0],           
            warps_per_cta=[4, 1],     
            instr_shape=[16, 64, 16], 
        )

        alpha_reg_layout: tl.constexpr = txl.SliceLayout(
            dim=1,            
            parent=mma_layout
        )

        for _n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mZ = txl.get_buffer(mZ, bufIdxR)
            txl.mbar_wait(cur_mZ, phase)
            cur_Z = txl.get_buffer(bZ, bufIdxR)
            cur_ZR = txl.smem_slice(cur_Z, R0//2, R0//2, 1)
            
            ##### bug place #####
            txl.mbar_wait(txl.get_buffer(bar_SS_ready, bufIdxR), phase)

            p = txl.get_buffer(S_shared, bufIdxR)
            alpha = txl.get_buffer(ALPHA_sh, bufIdxR)

            alpha_reg = txl.smem_load(alpha, alpha_reg_layout)
            accR = accR * alpha_reg[:, None]
            # accR = accR * alpha[:, None]
            accR = tl.dot(p, cur_ZR, accR)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(bar_SS_free, bufIdxR))
            txl.mbar_arrive(txl.get_buffer(mPV_R, bufIdxR))
            ##### bug place #####

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        txl.mbar_wait(txl.get_buffer(bar_sumexp_ready, 0), 0)
        l_i = txl.get_buffer(SUMEXP_sh, 0)
        accR = accR / l_i[:, None]
        # reg -> smem -> gmem
        txl.smem_store(baccR_Output, accR.to(dtype))
        txl.tma_store(baccR_Output, desc_o_lat, [qo_off, R0//2])

ws_cuta_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)  # 可按需调优
@txl.autotune(
    configs=[txl.Config(ws_cuta_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook)],
    key=["KV_SEQ_LEN","R0","PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_ws_dim2_2K_txl( # cutedsl 这里要求KV_SEQ_LEN是BLOCK_N*2的整数倍，因为我这边还没有写奇数倍数的逻辑
    sm_scale, M,                          
    Z, H, KV_HEADS,
    desc_qhat,                            # [Z,H,N_Q,R0]
    desc_zkv,                             # [Z,KV_HEADS,KV_SEQ_LEN,R0]
    desc_o_lat,                           # [Z,H,N_Q,R0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,                   # Q_pe: [Z,H,N_Q,PE], K_pe: [Z,KV_HEADS,KV_SEQ_LEN,PE]
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= R0)
    tl.static_assert(R0 % 2 == 0)       
    dtype = tl.float16

    pid_m = tl.program_id(0)        
    off_hz = tl.program_id(1)          
    off_z = off_hz // H
    off_h = off_hz %  H

    rows_q = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + pid_m * BLOCK_M
    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    bQ = txl.smem_alloc([BLOCK_M, R0], dtype=dtype)
    bQP  = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mQ = txl.mbar_alloc(1)
    mQP = txl.mbar_alloc(1)

    bZ0 = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    bZ1 = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    bKP0 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bKP1 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mZ0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mZ1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mKP0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mKP1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    mQK0 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mQK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV0_L = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV0_R = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV1_L = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV1_R = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    P0_shared = txl.smem_alloc([BLOCK_M, BLOCK_N], dtype=dtype, num_stages=NUM_STAGES)
    P1_shared = txl.smem_alloc([BLOCK_M, BLOCK_N], dtype=dtype, num_stages=NUM_STAGES)
    MAX0_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES) 
    MAX1_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES) 

    bar_SS0_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)  
    bar_SS0_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_SS1_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_SS1_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_Max0_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_Max0_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_Max1_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES) 
    bar_Max1_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    if txl.is_warpgroup([0]):
        bQi = txl.get_buffer(bQ, 0)
        mQi = txl.get_buffer(mQ, 0)
        bQPi = txl.get_buffer(bQP,  0)
        mQPi = txl.get_buffer(mQP,  0)

        txl.mbar_expect(mQi, BLOCK_M * R0 * 2)
        txl.mbar_expect(mQPi, BLOCK_M * PE_DIM * 2)

        txl.tma_load(bQi, desc_qhat, [qo_off, 0], mQi) 
        txl.mbar_wait(mQi, 0)
        txl.tma_load(bQPi,  desc_qpe, [qo_off, 0], mQPi)
        txl.mbar_wait(mQPi, 0)

        bufIdxW, phase = 0, 1
        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            cur_mZ0 = txl.get_buffer(mZ0, bufIdxW)
            cur_bZ0 = txl.get_buffer(bZ0, bufIdxW)
            cur_bKP0 = txl.get_buffer(bKP0, bufIdxW)
            cur_mKP0 = txl.get_buffer(mKP0, bufIdxW)

            cur_mZ1 = txl.get_buffer(mZ1, bufIdxW)
            cur_bZ1 = txl.get_buffer(bZ1, bufIdxW)
            cur_bKP1 = txl.get_buffer(bKP1, bufIdxW)
            cur_mKP1 = txl.get_buffer(mKP1, bufIdxW)

            txl.mbar_wait(txl.get_buffer(mQK0, bufIdxW), phase)
            txl.mbar_expect(cur_mKP0, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP0, desc_kpe, [kv_off, 0], cur_mKP0)

            txl.mbar_wait(txl.get_buffer(mQK1, bufIdxW), phase)
            txl.mbar_expect(cur_mKP1, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP1, desc_kpe, [kv_off+BLOCK_N, 0], cur_mKP1)

            txl.mbar_wait(txl.get_buffer(mPV0_L, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(mPV0_R, bufIdxW), phase)
            txl.mbar_expect(cur_mZ0, BLOCK_N*R0*2)
            txl.tma_load(cur_bZ0, desc_zkv, [kv_off, 0], cur_mZ0)

            txl.mbar_wait(txl.get_buffer(mPV1_L, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(mPV1_R, bufIdxW), phase)
            txl.mbar_expect(cur_mZ1, BLOCK_N*R0*2)
            txl.tma_load(cur_bZ1, desc_zkv, [kv_off+BLOCK_N, 0], cur_mZ1)

            kv_off += 2*BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase ^= 1
            
    if txl.is_warpgroup([1]):  # left consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accL= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)
        # accL_Output = txl.smem_alloc([BLOCK_M, R0//2], dtype=dtype)
        # baccL_Output = txl.get_buffer(accL_Output, 0)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mQP, 0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bQPi = txl.get_buffer(bQP,  0)

        bQli = txl.smem_slice(bQi, 0, R0//2, 1)
        bQri = txl.smem_slice(bQi, R0//2, R0//2, 1)

        bufIdxR, phase = 0, 0

        blocked_layout: tl.constexpr = txl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[1, 32],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )

        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            cur_mKP0 = txl.get_buffer(mKP0, bufIdxR)
            txl.mbar_wait(cur_mKP0, phase)
            cur_KP0 = txl.get_buffer(bKP0, bufIdxR)

            acc_s = tl.dot(bQPi, cur_KP0.T)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(mQK0, bufIdxR))

            cur_mZ0 = txl.get_buffer(mZ0, bufIdxR)
            txl.mbar_wait(cur_mZ0, phase)
            cur_Z0 = txl.get_buffer(bZ0, bufIdxR)
            cur_ZL0 = txl.smem_slice(cur_Z0, 0, R0//2, 1)
            cur_ZR0 = txl.smem_slice(cur_Z0, R0//2, R0//2, 1)

            acc_s += tl.dot(bQli, cur_ZL0.T)
            txl.dot_wait(0)
            acc_s += tl.dot(bQri, cur_ZR0.T)
            txl.dot_wait(0)

            m_ij0  = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
            txl.mbar_wait(txl.get_buffer(bar_Max0_free, bufIdxR), phase^1)
            txl.smem_store(txl.get_buffer(MAX0_sh, bufIdxR), m_ij0)
            txl.mbar_arrive(txl.get_buffer(bar_Max0_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_Max1_ready, bufIdxR), phase)
            m_ij1 = txl.get_buffer(MAX1_sh, bufIdxR)
            m_ij = tl.maximum(m_ij0, m_ij1)
            txl.mbar_arrive(txl.get_buffer(bar_Max1_free, bufIdxR))

            acc_s = acc_s * qk_scale - m_ij[:, None]
            p0 = tl.math.exp2(acc_s)
            l_ij0  = tl.sum(p0, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij0
            accL  = accL * alpha[:, None]
            m_i   = m_ij

            txl.mbar_wait(txl.get_buffer(bar_SS0_free, bufIdxR), phase^1)
            cur_P0_shared = txl.get_buffer(P0_shared, bufIdxR)
            txl.smem_store(cur_P0_shared, p0.to(dtype))
            txl.mbar_arrive(txl.get_buffer(bar_SS0_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_SS1_ready, bufIdxR), phase)
            p1 = txl.get_buffer(P1_shared, bufIdxR)

            p1_reg = txl.smem_load(p1, blocked_layout)

            l_ij1  = tl.sum(p1_reg, 1)
            l_i  = l_i + l_ij1

            accL = tl.dot(p0.to(dtype), cur_ZL0, accL)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(mPV0_L, bufIdxR))

            cur_mZ1 = txl.get_buffer(mZ1, bufIdxR)
            txl.mbar_wait(cur_mZ1, phase)
            cur_Z1 = txl.get_buffer(bZ1, bufIdxR)

            cur_ZL1 = txl.smem_slice(cur_Z1, 0, R0//2, 1)

            accL = tl.dot(p1.to(dtype), cur_ZL1, accL)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(bar_SS1_free, bufIdxR))
            txl.mbar_arrive(txl.get_buffer(mPV1_L, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        m_i += tl.math.log2(l_i)
        accL = accL / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i) 
        # reg -> smem -> gmem
        baccL_Output = txl.smem_slice(txl.get_buffer(bZ1, 0), 0, R0//2, 1)
        txl.smem_store(baccL_Output, accL.to(dtype))
        txl.tma_store(baccL_Output, desc_o_lat, [qo_off, 0])            

    if txl.is_warpgroup([2]):  # right consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accR= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)
        # accR_Output = txl.smem_alloc([BLOCK_M, R0//2], dtype=dtype)
        # baccR_Output = txl.get_buffer(accR_Output, 0)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mQP, 0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bQPi = txl.get_buffer(bQP,  0)

        bQli = txl.smem_slice(bQi, 0, R0//2, 1)
        bQri = txl.smem_slice(bQi, R0//2, R0//2, 1)
        bufIdxR, phase = 0, 0

        blocked_layout: tl.constexpr = txl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[1, 32],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        
        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            cur_mKP1 = txl.get_buffer(mKP1, bufIdxR)
            txl.mbar_wait(cur_mKP1, phase)
            cur_KP1 = txl.get_buffer(bKP1, bufIdxR)

            acc_s = tl.dot(bQPi, cur_KP1.T)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(mQK1, bufIdxR))

            cur_mZ1 = txl.get_buffer(mZ1, bufIdxR)
            txl.mbar_wait(cur_mZ1, phase)
            cur_Z1 = txl.get_buffer(bZ1, bufIdxR)

            cur_ZL1 = txl.smem_slice(cur_Z1, 0, R0//2, 1)
            cur_ZR1 = txl.smem_slice(cur_Z1, R0//2, R0//2, 1)

            acc_s += tl.dot(bQli, cur_ZL1.T)
            txl.dot_wait(0)
            acc_s += tl.dot(bQri, cur_ZR1.T)
            txl.dot_wait(0)
            
            m_ij1  = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
            txl.mbar_wait(txl.get_buffer(bar_Max1_free, bufIdxR), phase^1)
            txl.smem_store(txl.get_buffer(MAX1_sh, bufIdxR), m_ij1)
            txl.mbar_arrive(txl.get_buffer(bar_Max1_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_Max0_ready, bufIdxR), phase)
            m_ij0 = txl.get_buffer(MAX0_sh, bufIdxR)
            m_ij = tl.maximum(m_ij0, m_ij1)
            txl.mbar_arrive(txl.get_buffer(bar_Max0_free, bufIdxR))

            acc_s = acc_s * qk_scale - m_ij[:, None]
            p1 = tl.math.exp2(acc_s)
            l_ij1  = tl.sum(p1, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij1
            accR  = accR * alpha[:, None]
            m_i   = m_ij

            txl.mbar_wait(txl.get_buffer(bar_SS1_free, bufIdxR), phase^1)
            cur_P1_shared = txl.get_buffer(P1_shared, bufIdxR)
            txl.smem_store(cur_P1_shared, p1.to(dtype))
            txl.mbar_arrive(txl.get_buffer(bar_SS1_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_SS0_ready, bufIdxR), phase)
            p0 = txl.get_buffer(P0_shared, bufIdxR)

            p0_reg = txl.smem_load(p0, blocked_layout)

            l_ij0  = tl.sum(p0_reg, 1)
            l_i  = l_i + l_ij0

            accR = tl.dot(p1.to(dtype), cur_ZR1, accR)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(mPV1_R, bufIdxR))

            cur_mZ0 = txl.get_buffer(mZ0, bufIdxR)
            txl.mbar_wait(cur_mZ0, phase)
            cur_Z0 = txl.get_buffer(bZ0, bufIdxR)
            cur_ZR0 = txl.smem_slice(cur_Z0, R0//2, R0//2, 1)

            accR = tl.dot(p0.to(dtype), cur_ZR0, accR)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(bar_SS0_free, bufIdxR))
            txl.mbar_arrive(txl.get_buffer(mPV0_R, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        accR = accR / l_i[:, None]
        baccR_Output = txl.smem_slice(txl.get_buffer(bZ1, 0), 0, R0//2, 1)
        txl.smem_store(baccR_Output, accR.to(dtype))
        txl.tma_store(baccR_Output, desc_o_lat, [qo_off, R0//2])

ws_cuta_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)  # 可按需调优
@txl.autotune(
    configs=[txl.Config(ws_cuta_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook)],
    key=["KV_SEQ_LEN","R0","PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_ws_dim2_2K_txl_debug( # cutedsl 这里要求KV_SEQ_LEN是BLOCK_N*2的整数倍，因为我这边还没有写奇数倍数的逻辑
    sm_scale, M,                          
    Z, H, KV_HEADS,
    desc_qhat,                            # [Z,H,N_Q,R0]
    desc_zkv,                             # [Z,KV_HEADS,KV_SEQ_LEN,R0]
    desc_o_lat,                           # [Z,H,N_Q,R0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,                   # Q_pe: [Z,H,N_Q,PE], K_pe: [Z,KV_HEADS,KV_SEQ_LEN,PE]
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= R0)
    tl.static_assert(R0 % 2 == 0)
    tl.static_assert(PE_DIM == BLOCK_N)  
    tl.static_assert(NUM_STAGES == 1)
    dtype = tl.float16

    pid_m = tl.program_id(0)        
    off_hz = tl.program_id(1)          
    off_z = off_hz // H
    off_h = off_hz %  H

    rows_q = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + pid_m * BLOCK_M
    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    bQ = txl.smem_alloc([BLOCK_M, R0], dtype=dtype)
    bQP  = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mQ = txl.mbar_alloc(1)
    mQP = txl.mbar_alloc(1)

    bZ0 = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    bZ1 = txl.smem_alloc([BLOCK_N, R0], dtype=dtype, num_stages=NUM_STAGES)
    bKP0 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    bKP1 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=NUM_STAGES)
    mZ0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mZ1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mKP0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mKP1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)

    mQK0 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mQK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV0_L = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV0_R = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV1_L = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mPV1_R = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    P0_shared = txl.smem_alloc([BLOCK_M, BLOCK_N], dtype=dtype, num_stages=NUM_STAGES)
    P1_shared = bQP
    MAX0_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES) 
    MAX1_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES) 
    L0_shared = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES)
    L1_shared = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=NUM_STAGES)

    bar_SS0_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)  
    bar_SS0_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_SS1_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_SS1_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_Max0_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_Max0_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_Max1_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES) 
    bar_Max1_free  = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    bar_L0_ready = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    bar_L1_ready  = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    bar_bQP_free = txl.mbar_alloc(128, num_stages=NUM_STAGES)

    mma_layout: tl.constexpr = txl.NVMMADistributedLayout(
        version=[3, 0],           
        warps_per_cta=[4, 1],     
        instr_shape=[16, 64, 16], 
    )

    max_reg_layout: tl.constexpr = txl.SliceLayout(
            dim=1,            
            parent=mma_layout
    )

    A_op_layout: tl.constexpr = txl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)

    if txl.is_warpgroup([0]):
        bQi = txl.get_buffer(bQ, 0)
        mQi = txl.get_buffer(mQ, 0)
        bQPi = txl.get_buffer(bQP,  0)
        mQPi = txl.get_buffer(mQP,  0)

        txl.mbar_expect(mQi, BLOCK_M * R0 * 2)
        txl.mbar_expect(mQPi, BLOCK_M * PE_DIM * 2)

        txl.tma_load(bQi, desc_qhat, [qo_off, 0], mQi) 
        txl.mbar_wait(mQi, 0)
        txl.tma_load(bQPi,  desc_qpe, [qo_off, 0], mQPi)
        txl.mbar_wait(mQPi, 0)

        bufIdxW, phase = 0, 1
        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            cur_mZ0 = txl.get_buffer(mZ0, bufIdxW)
            cur_bZ0 = txl.get_buffer(bZ0, bufIdxW)
            cur_bKP0 = txl.get_buffer(bKP0, bufIdxW)
            cur_mKP0 = txl.get_buffer(mKP0, bufIdxW)

            cur_mZ1 = txl.get_buffer(mZ1, bufIdxW)
            cur_bZ1 = txl.get_buffer(bZ1, bufIdxW)
            cur_bKP1 = txl.get_buffer(bKP1, bufIdxW)
            cur_mKP1 = txl.get_buffer(mKP1, bufIdxW)

            txl.mbar_wait(txl.get_buffer(mQK0, bufIdxW), phase)
            txl.mbar_expect(cur_mKP0, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP0, desc_kpe, [kv_off, 0], cur_mKP0)

            txl.mbar_wait(txl.get_buffer(mQK1, bufIdxW), phase)
            txl.mbar_expect(cur_mKP1, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP1, desc_kpe, [kv_off+BLOCK_N, 0], cur_mKP1)

            txl.mbar_wait(txl.get_buffer(mPV0_L, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(mPV0_R, bufIdxW), phase)
            txl.mbar_expect(cur_mZ0, BLOCK_N*R0*2)
            txl.tma_load(cur_bZ0, desc_zkv, [kv_off, 0], cur_mZ0)

            txl.mbar_wait(txl.get_buffer(mPV1_L, bufIdxW), phase)
            txl.mbar_wait(txl.get_buffer(mPV1_R, bufIdxW), phase)
            txl.mbar_expect(cur_mZ1, BLOCK_N*R0*2)
            txl.tma_load(cur_bZ1, desc_zkv, [kv_off+BLOCK_N, 0], cur_mZ1)

            kv_off += 2*BLOCK_N
            bufIdxW = (bufIdxW + 1) % NUM_STAGES
            if bufIdxW == 0:
                phase ^= 1
            
    if txl.is_warpgroup([1]):  # left consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accL= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mQP, 0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bQPi = txl.get_buffer(bQP,  0)
        rQP = txl.smem_load(bQPi, A_op_layout)
        # txl.mbar_arrive(txl.get_buffer(bar_bQP_free, 0))

        bQli = txl.smem_slice(bQi, 0, R0//2, 1)
        bQri = txl.smem_slice(bQi, R0//2, R0//2, 1)

        bufIdxR, phase = 0, 0

        blocked_layout: tl.constexpr = txl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[1, 32],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )

        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            cur_mKP0 = txl.get_buffer(mKP0, bufIdxR)
            txl.mbar_wait(cur_mKP0, phase)
            cur_KP0 = txl.get_buffer(bKP0, bufIdxR)

            acc_s = tl.dot(rQP, cur_KP0.T)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(mQK0, bufIdxR))

            cur_mZ0 = txl.get_buffer(mZ0, bufIdxR)
            txl.mbar_wait(cur_mZ0, phase)
            cur_Z0 = txl.get_buffer(bZ0, bufIdxR)
            cur_ZL0 = txl.smem_slice(cur_Z0, 0, R0//2, 1)
            cur_ZR0 = txl.smem_slice(cur_Z0, R0//2, R0//2, 1)

            acc_s += tl.dot(bQli, cur_ZL0.T)
            txl.dot_wait(0)
            acc_s += tl.dot(bQri, cur_ZR0.T)
            txl.dot_wait(0)

            m_ij0  = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
            txl.mbar_wait(txl.get_buffer(bar_Max0_free, bufIdxR), phase^1)
            txl.smem_store(txl.get_buffer(MAX0_sh, bufIdxR), m_ij0)
            txl.mbar_arrive(txl.get_buffer(bar_Max0_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_Max1_ready, bufIdxR), phase)
            m_ij1 = txl.get_buffer(MAX1_sh, bufIdxR)
            m_ij1_reg = txl.smem_load(m_ij1,max_reg_layout)
            m_ij = tl.maximum(m_ij0, m_ij1_reg)
            txl.mbar_arrive(txl.get_buffer(bar_Max1_free, bufIdxR))

            acc_s = acc_s * qk_scale - m_ij[:, None]
            p0 = tl.math.exp2(acc_s)
            l_ij0  = tl.sum(p0, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij0
            accL  = accL * alpha[:, None]
            m_i   = m_ij

            txl.mbar_wait(txl.get_buffer(bar_SS0_free, bufIdxR), phase^1)
            cur_P0_shared = txl.get_buffer(P0_shared, bufIdxR)
            txl.smem_store(cur_P0_shared, p0.to(dtype))
            txl.mbar_arrive(txl.get_buffer(bar_SS0_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_SS1_ready, bufIdxR), phase)
            p1 = txl.get_buffer(P1_shared, bufIdxR)

            accL = tl.dot(p0.to(dtype), cur_ZL0, accL)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(mPV0_L, bufIdxR))

            cur_mZ1 = txl.get_buffer(mZ1, bufIdxR)
            txl.mbar_wait(cur_mZ1, phase)
            cur_Z1 = txl.get_buffer(bZ1, bufIdxR)

            cur_ZL1 = txl.smem_slice(cur_Z1, 0, R0//2, 1)

            accL = tl.dot(p1.to(dtype), cur_ZL1, accL)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(bar_SS1_free, bufIdxR))
            txl.mbar_arrive(txl.get_buffer(mPV1_L, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        L0_sh = txl.get_buffer(L0_shared, 0)
        txl.smem_store(L0_sh, l_i)
        txl.mbar_arrive(txl.get_buffer(bar_L0_ready, 0))
        txl.mbar_wait(txl.get_buffer(bar_L1_ready, 0), 0)
        L1_sh = txl.get_buffer(L1_shared, 0)
        l_i = l_i + L1_sh
        m_i += tl.math.log2(l_i)
        accL = accL / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i) 
        # reg -> smem -> gmem
        baccL_Output = txl.smem_slice(txl.get_buffer(bZ0, 0), 0, R0//2, 1)
        txl.smem_store(baccL_Output, accL.to(dtype))
        txl.tma_store(baccL_Output, desc_o_lat, [qo_off, 0])            

    if txl.is_warpgroup([2]):  # right consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accR= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)

        txl.mbar_wait(txl.get_buffer(mQ, 0), 0)
        txl.mbar_wait(txl.get_buffer(mQP, 0), 0)
        bQi = txl.get_buffer(bQ, 0)
        bQPi = txl.get_buffer(bQP,  0)
        rQP = txl.smem_load(bQPi, A_op_layout)
        # txl.mbar_wait(txl.get_buffer(bar_bQP_free, 0), 0)

        bQli = txl.smem_slice(bQi, 0, R0//2, 1)
        bQri = txl.smem_slice(bQi, R0//2, R0//2, 1)
        bufIdxR, phase = 0, 0

        blocked_layout: tl.constexpr = txl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[1, 32],
            warps_per_cta=[2, 2],
            order=[1, 0],
        )
        
        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            cur_mKP1 = txl.get_buffer(mKP1, bufIdxR)
            txl.mbar_wait(cur_mKP1, phase)
            cur_KP1 = txl.get_buffer(bKP1, bufIdxR)

            acc_s = tl.dot(rQP, cur_KP1.T)
            txl.dot_wait(0)

            txl.mbar_arrive(txl.get_buffer(mQK1, bufIdxR))

            cur_mZ1 = txl.get_buffer(mZ1, bufIdxR)
            txl.mbar_wait(cur_mZ1, phase)
            cur_Z1 = txl.get_buffer(bZ1, bufIdxR)

            cur_ZL1 = txl.smem_slice(cur_Z1, 0, R0//2, 1)
            cur_ZR1 = txl.smem_slice(cur_Z1, R0//2, R0//2, 1)

            acc_s += tl.dot(bQli, cur_ZL1.T)
            txl.dot_wait(0)
            acc_s += tl.dot(bQri, cur_ZR1.T)
            txl.dot_wait(0)
            
            m_ij1  = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
            txl.mbar_wait(txl.get_buffer(bar_Max1_free, bufIdxR), phase^1)
            txl.smem_store(txl.get_buffer(MAX1_sh, bufIdxR), m_ij1)
            txl.mbar_arrive(txl.get_buffer(bar_Max1_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_Max0_ready, bufIdxR), phase)
            m_ij0 = txl.get_buffer(MAX0_sh, bufIdxR)
            m_ij0_reg = txl.smem_load(m_ij0, max_reg_layout)
            m_ij = tl.maximum(m_ij0_reg, m_ij1)
            txl.mbar_arrive(txl.get_buffer(bar_Max0_free, bufIdxR))

            acc_s = acc_s * qk_scale - m_ij[:, None]
            p1 = tl.math.exp2(acc_s)
            l_ij1  = tl.sum(p1, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij1
            accR  = accR * alpha[:, None]
            m_i   = m_ij

            txl.mbar_wait(txl.get_buffer(bar_SS1_free, bufIdxR), phase^1)
            cur_P1_shared = txl.get_buffer(P1_shared, bufIdxR)
            txl.smem_store(cur_P1_shared, p1.to(dtype))
            txl.mbar_arrive(txl.get_buffer(bar_SS1_ready, bufIdxR))

            txl.mbar_wait(txl.get_buffer(bar_SS0_ready, bufIdxR), phase)
            p0 = txl.get_buffer(P0_shared, bufIdxR)

            accR = tl.dot(p1.to(dtype), cur_ZR1, accR)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(mPV1_R, bufIdxR))

            cur_mZ0 = txl.get_buffer(mZ0, bufIdxR)
            txl.mbar_wait(cur_mZ0, phase)
            cur_Z0 = txl.get_buffer(bZ0, bufIdxR)
            cur_ZR0 = txl.smem_slice(cur_Z0, R0//2, R0//2, 1)

            accR = tl.dot(p0.to(dtype), cur_ZR0, accR)
            txl.dot_wait(0)
            txl.mbar_arrive(txl.get_buffer(bar_SS0_free, bufIdxR))
            txl.mbar_arrive(txl.get_buffer(mPV0_R, bufIdxR))

            bufIdxR = (bufIdxR + 1) % NUM_STAGES
            if bufIdxR == 0:
                phase ^= 1

        L1_sh = txl.get_buffer(L1_shared, 0)
        txl.smem_store(L1_sh, l_i)
        txl.mbar_arrive(txl.get_buffer(bar_L1_ready, 0))
        txl.mbar_wait(txl.get_buffer(bar_L0_ready, 0), 0)
        L0_sh = txl.get_buffer(L0_shared, 0)
        l_i = l_i + L0_sh
        accR = accR / l_i[:, None]
        baccR_Output = txl.smem_slice(txl.get_buffer(bZ1, 0), 0, R0//2, 1)
        txl.smem_store(baccR_Output, accR.to(dtype))
        txl.tma_store(baccR_Output, desc_o_lat, [qo_off, R0//2])

def _pre_hook(nargs):
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
ws_cuta_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)  # 可按需调优
@txl.autotune(
    configs=[txl.Config(ws_cuta_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook)],
    key=["KV_SEQ_LEN","R0","PE_DIM"]
)
@txl.jit
def mla_decode_latent_sharedZ_ws_dim2_2K_txl_change( # cutedsl 这里要求KV_SEQ_LEN是BLOCK_N*2的整数倍，因为我这边还没有写奇数倍数的逻辑
    sm_scale, M,                          
    Z, H, KV_HEADS,
    desc_qhat,                            # [Z,H,N_Q,R0]
    desc_zkv,                             # [Z,KV_HEADS,KV_SEQ_LEN,R0]
    desc_o_lat,                           # [Z,H,N_Q,R0]
    N_Q, KV_SEQ_LEN,
    desc_qpe, desc_kpe,                   # Q_pe: [Z,H,N_Q,PE], K_pe: [Z,KV_HEADS,KV_SEQ_LEN,PE]
    PE_DIM: tl.constexpr,
    R0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= R0)
    tl.static_assert(R0 % 2 == 0)
    tl.static_assert(PE_DIM == BLOCK_N)  
    tl.static_assert(NUM_STAGES == 1)
    dtype = tl.float16
    tid = txl.tid(0)
    idx_in_warpgroup = tid % 128

    pid_m = tl.program_id(0)        
    off_hz = tl.program_id(1)          
    off_z = off_hz // H
    off_h = off_hz %  H

    rows_q = Z * H * N_Q
    rows_kv = Z * KV_HEADS * KV_SEQ_LEN

    desc_qhat = _maybe_desc(desc_qhat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, R0], [R0, 1], [BLOCK_M, R0])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, R0], [R0, 1], [BLOCK_N, R0])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])

    q_base = off_z * (H * N_Q) + off_h * N_Q
    qo_off = q_base + pid_m * BLOCK_M
    heads_per_kv = H // KV_HEADS
    kv_head_idx = off_h // heads_per_kv
    kv_base = off_z * (KV_HEADS * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
    kv_off = kv_base

    bQ = txl.smem_alloc([BLOCK_M, R0], dtype=dtype)
    bQP = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mQ = txl.mbar_alloc(1)
    mQP = txl.mbar_alloc(1)

    bZL0 = txl.smem_alloc([BLOCK_N, R0//2], dtype=dtype, num_stages=1)
    bZR0 = txl.smem_alloc([BLOCK_N, R0//2], dtype=dtype, num_stages=1)
    bZL1 = txl.smem_alloc([BLOCK_N, R0//2], dtype=dtype, num_stages=1)
    bZR1 = txl.smem_alloc([BLOCK_N, R0//2], dtype=dtype, num_stages=1)
    bKP0 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=1)
    bKP1 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=1)
    mZL0 = txl.mbar_alloc(1, num_stages=1)
    mZR0 = txl.mbar_alloc(1, num_stages=1)
    mZL1 = txl.mbar_alloc(1, num_stages=1)
    mZR1 = txl.mbar_alloc(1, num_stages=1)
    mKP0 = txl.mbar_alloc(1, num_stages=1)
    mKP1 = txl.mbar_alloc(1, num_stages=1)

    mQK0 = txl.mbar_alloc(128, num_stages=1)
    mQK1 = txl.mbar_alloc(128, num_stages=1)
    mPV0_L = txl.mbar_alloc(128, num_stages=1)
    mPV0_R = txl.mbar_alloc(128, num_stages=1)
    mPV1_L = txl.mbar_alloc(128, num_stages=1)
    mPV1_R = txl.mbar_alloc(128, num_stages=1)

    P0_sh = txl.smem_alloc([BLOCK_M, BLOCK_N], dtype=dtype, num_stages=1)
    P1_sh = bQP
    Max_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=1)
    L0_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=1)
    L1_sh = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=1)


    bar_P0_ready = txl.mbar_alloc(128, num_stages=1)  
    bar_P1_ready = txl.mbar_alloc(128, num_stages=1)
    bar_Max0_ready = txl.mbar_alloc(128, num_stages=1)
    bar_Max1_ready = txl.mbar_alloc(128, num_stages=1) 

    bar_L0_ready = txl.mbar_alloc(128, num_stages=1)
    bar_L1_ready  = txl.mbar_alloc(128, num_stages=1)

    mma_layout: tl.constexpr = txl.NVMMADistributedLayout(
        version=[3, 0],           
        warps_per_cta=[4, 1],     
        instr_shape=[16, 64, 16], 
    )

    max_reg_layout: tl.constexpr = txl.SliceLayout(
            dim=1,            
            parent=mma_layout
    )

    A_op_layout: tl.constexpr = txl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)

    cur_bQ = txl.get_buffer(bQ, 0)
    cur_mQ = txl.get_buffer(mQ, 0)
    cur_bQP = txl.get_buffer(bQP,  0)
    cur_mQP = txl.get_buffer(mQP,  0)

    cur_mZL0 = txl.get_buffer(mZL0, 0)
    cur_mZR0 = txl.get_buffer(mZR0, 0)
    cur_bZL0 = txl.get_buffer(bZL0, 0)
    cur_bZR0 = txl.get_buffer(bZR0, 0)
    cur_bKP0 = txl.get_buffer(bKP0, 0)
    cur_mKP0 = txl.get_buffer(mKP0, 0)

    cur_mZL1 = txl.get_buffer(mZL1, 0)
    cur_mZR1 = txl.get_buffer(mZR1, 0)
    cur_bZL1 = txl.get_buffer(bZL1, 0)
    cur_bZR1 = txl.get_buffer(bZR1, 0)
    cur_bKP1 = txl.get_buffer(bKP1, 0)
    cur_mKP1 = txl.get_buffer(mKP1, 0)

    cur_mQK0 = txl.get_buffer(mQK0, 0)
    cur_mQK1 = txl.get_buffer(mQK1, 0)
    cur_mPV0_L = txl.get_buffer(mPV0_L, 0)
    cur_mPV0_R = txl.get_buffer(mPV0_R, 0)
    cur_mPV1_L = txl.get_buffer(mPV1_L, 0)
    cur_mPV1_R = txl.get_buffer(mPV1_R, 0)

    # TODO: 初始化cur_Max_sh，每次循环置零
    cur_Max_sh = txl.get_buffer(Max_sh, 0)
    cur_P0_sh = txl.get_buffer(P0_sh, 0)
    cur_P1_sh = txl.get_buffer(P1_sh, 0)
    cur_L0_sh = txl.get_buffer(L0_sh, 0)
    cur_L1_sh = txl.get_buffer(L1_sh, 0)

    cur_Max0_ready = txl.get_buffer(bar_Max0_ready, 0)
    cur_Max1_ready = txl.get_buffer(bar_Max1_ready, 0)
    cur_L0_ready = txl.get_buffer(bar_L0_ready, 0)
    cur_L1_ready = txl.get_buffer(bar_L1_ready, 0)
    cur_P0_ready = txl.get_buffer(bar_P0_ready, 0)
    cur_P1_ready = txl.get_buffer(bar_P1_ready, 0)

    if txl.is_warpgroup([0]):

        txl.mbar_expect(cur_mQ, BLOCK_M * R0 * 2)
        txl.mbar_expect(cur_mQP, BLOCK_M * PE_DIM * 2)

        txl.tma_load(cur_bQ, desc_qhat, [qo_off, 0], cur_mQ) 
        txl.mbar_wait(cur_mQ, 0)
        txl.tma_load(cur_bQP,  desc_qpe, [qo_off, 0], cur_mQP)
        txl.mbar_wait(cur_mQP, 0)

        phase = 1
        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):

            txl.mbar_wait(cur_mQK0, phase)
            txl.mbar_expect(cur_mKP0, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP0, desc_kpe, [kv_off, 0], cur_mKP0)

            txl.mbar_wait(cur_mQK1, phase)
            txl.mbar_expect(cur_mKP1, BLOCK_N*PE_DIM*2)
            txl.tma_load(cur_bKP1, desc_kpe, [kv_off+BLOCK_N, 0], cur_mKP1)

            txl.mbar_wait(cur_mPV0_L, phase)
            txl.mbar_expect(cur_mZL0, BLOCK_N*R0//2*2)
            txl.tma_load(cur_bZL0, desc_zkv, [kv_off, 0], cur_mZL0)

            txl.mbar_wait(cur_mPV1_R, phase)
            txl.mbar_expect(cur_mZR1, BLOCK_N*R0//2*2)
            txl.tma_load(cur_bZR1, desc_zkv, [kv_off+BLOCK_N, R0//2], cur_mZR1)

            txl.mbar_wait(cur_mPV0_R, phase)
            txl.mbar_expect(cur_mZR0, BLOCK_N*R0//2*2)
            txl.tma_load(cur_bZR0, desc_zkv, [kv_off, R0//2], cur_mZR0)

            txl.mbar_wait(cur_mPV1_L, phase)
            txl.mbar_expect(cur_mZL1, BLOCK_N*R0//2*2)
            txl.tma_load(cur_bZL1, desc_zkv, [kv_off+BLOCK_N, 0], cur_mZL1)

            kv_off += 2*BLOCK_N
            phase ^= 1
            
    if txl.is_warpgroup([1]):  # left consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32) 
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accL= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)

        txl.mbar_wait(cur_mQ, 0)
        txl.mbar_wait(cur_mQP, 0)
        rQP = txl.smem_load(cur_bQP, A_op_layout)
        txl.print("rQP:", rQP)

        cur_bQl = txl.smem_slice(cur_bQ, 0, R0//2, 1)
        cur_bQr = txl.smem_slice(cur_bQ, R0//2, R0//2, 1)

        phase = 0

        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):
            txl.mbar_wait(cur_mKP0, phase)
            acc_s = tl.dot(rQP, cur_bKP0.T)
            txl.dot_wait(0)

            txl.mbar_arrive(cur_mQK0)

            # TODO: slice pipeline
            txl.mbar_wait(cur_mZL0, phase)
            acc_s += tl.dot(cur_bQl, cur_bZL0.T)
            txl.dot_wait(0)

            txl.mbar_wait(cur_mZR0, phase)
            acc_s += tl.dot(cur_bQr, cur_bZR0.T)
            txl.dot_wait(0)

            m_ij0  = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
            alpha0 = tl.math.exp2(m_i - m_ij0)

            if idx_in_warpgroup % 4 == 0:
                txl.smem_store(cur_Max_sh, m_ij0)
            txl.mbar_arrive(cur_Max0_ready)

            acc_s = acc_s * qk_scale - m_ij0[:, None]
            p0 = tl.math.exp2(acc_s)
            l_ij0 = tl.sum(p0, 1)
            l_i = l_i * alpha0 + l_ij0
            accL = accL * alpha0[:, None]
            m_i = m_ij0
            accL = tl.dot(p0.to(dtype), cur_bZL0, accL)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mPV0_L)

            txl.mbar_wait(cur_Max1_ready, phase)
            m_ij1 = txl.frag_smem_load(cur_Max_sh, [64], max_reg_layout)
            alpha1 = tl.math.exp2(m_i - m_ij1)
            m_i = m_ij1

            # rescale P0
            p0 = p0 * alpha1[:, None]

            txl.smem_store(cur_P0_sh, p0.to(dtype))
            txl.mbar_arrive(cur_P0_ready)


            txl.mbar_wait(cur_mZL1, phase)

            # rescale accL
            accL = accL * alpha1[:, None]
            l_i = l_i * alpha1

            txl.mbar_wait(cur_P1_ready, phase)

            accL = tl.dot(cur_P1_sh.to(dtype), cur_bZL1, accL)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mPV1_L)

            phase ^= 1

        if idx_in_warpgroup % 4 == 0:
            txl.smem_store(cur_L0_sh, l_i)
        txl.mbar_arrive(cur_L0_ready)
        txl.mbar_wait(cur_L1_ready, 0)
        L1_reg = txl.frag_smem_load(cur_L1_sh, (64,), max_reg_layout)
        l_i = l_i + L1_reg
        # m_i = txl.frag_smem_load(cur_Max_sh, (64,), max_reg_layout)
        m_i += tl.math.log2(l_i)
        accL = accL / l_i[:, None]
        m_ptrs = M + off_hz * N_Q + offs_m
        tl.store(m_ptrs, m_i) 
        # reg -> smem -> gmem
        txl.smem_store(cur_bZL0, accL.to(dtype))
        txl.tma_store(cur_bZL0, desc_o_lat, [qo_off, 0])            

    if txl.is_warpgroup([2]):  # right consumer
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qk_scale = sm_scale * 1.44269504

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32) 
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
        accR= tl.zeros([BLOCK_M, R0//2], dtype=tl.float32)

        txl.mbar_wait(cur_mQ, 0)
        txl.mbar_wait(cur_mQP, 0)
        rQP = txl.smem_load(cur_bQP, A_op_layout)
        txl.print("rQP:", rQP)

        cur_bQl = txl.smem_slice(cur_bQ, 0, R0//2, 1)
        cur_bQr = txl.smem_slice(cur_bQ, R0//2, R0//2, 1)

        phase = 0
        
        for _n in range(0, KV_SEQ_LEN, BLOCK_N*2):

            txl.mbar_wait(cur_mKP1, phase)
            acc_s = tl.dot(rQP, cur_bKP1.T)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mQK1)

            # TODO: slice pipeline
            txl.mbar_wait(cur_mZR1, phase)
            acc_s += tl.dot(cur_bQr, cur_bZR1.T)
            txl.dot_wait(0)

            txl.mbar_wait(cur_mZL1, phase)
            acc_s += tl.dot(cur_bQl, cur_bZL1.T)
            txl.dot_wait(0)

            txl.mbar_wait(cur_Max0_ready, phase)
            m_ij0 = txl.frag_smem_load(cur_Max_sh, [64], max_reg_layout)
            
            m_ij1 = tl.maximum(m_ij0, tl.max(acc_s, 1) * qk_scale)
            alpha1 = tl.math.exp2(m_i - m_ij1)
            acc_s = acc_s * qk_scale - m_ij1[:, None]
            p1 = tl.math.exp2(acc_s)
            l_ij1 = tl.sum(p1, 1)
            l_i = l_i * alpha1 + l_ij1
            accR = accR * alpha1[:, None]
            m_i = m_ij1

            if idx_in_warpgroup % 4 == 0:
                txl.smem_store(cur_Max_sh, m_ij1)
            txl.mbar_arrive(cur_Max1_ready)

            accR = tl.dot(p1.to(dtype), cur_bZR1, accR)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mPV1_R)

            txl.mbar_wait(cur_P0_ready, phase)
            txl.mbar_wait(cur_mZR0, phase)
            accR = tl.dot(cur_P0_sh.to(dtype), cur_bZR0, accR)
            txl.dot_wait(0)
            txl.mbar_arrive(cur_mPV0_R)

            txl.smem_store(cur_P1_sh, p1.to(dtype))
            txl.mbar_arrive(cur_P1_ready)

            phase ^= 1

        if idx_in_warpgroup % 4 == 0:
            txl.smem_store(cur_L1_sh, l_i)
        txl.mbar_arrive(cur_L1_ready)
        txl.mbar_wait(cur_L0_ready, 0)
        L0_reg = txl.frag_smem_load(cur_L0_sh, (64,), max_reg_layout)
        l_i = l_i + L0_reg
        accR = accR / l_i[:, None]
        txl.smem_store(cur_bZL1, accR.to(dtype))
        txl.tma_store(cur_bZL1, desc_o_lat, [qo_off, R0//2])

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
    # desc_q = _maybe_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
    #                                  block_shape=[BLOCK_M, HEAD_DIM])
    # desc_v = _maybe_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
    #                                  block_shape=[BLOCK_N, HEAD_DIM])
    # desc_k = _maybe_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
    #                                  block_shape=[BLOCK_N, HEAD_DIM])
    # desc_o = _maybe_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
    #                                  block_shape=[BLOCK_M, HEAD_DIM])

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

def mla_test(q, kv, qpe, kpe, sm_scale, algo=0):
    HEAD_DIM_Q = q.shape[-1]
    HEAD_DIM_Z = kv.shape[-1]
    HEAD_DIM_PE = qpe.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_Z
    o = torch.empty_like(q)
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
        0: mla_decode_latent_sharedZ_txl,
        1: mla_decode_latent_sharedZ_ws_txl,
        2: mla_decode_latent_sharedZ_wsNsplit_txl,
        3: mla_decode_latent_sharedZ_ws_dim2_txl,
        4: mla_decode_latent_sharedZ_ws_dim2_2K_txl,
        5: mla_decode_latent_sharedZ_wsNsplit_txl_debug,
        6: mla_decode_latent_sharedZ_ws_dim2_2K_txl_debug,
        7: mla_decode_latent_sharedZ_ws_dim2_2K_txl_change,
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

def debug_test(q, kv, qpe, kpe, v, sm_scale, algo=0):
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
        desc_v = TensorDescriptor(v, shape=[kv_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
    else:
        desc_q = q
        desc_kv = kv
        desc_kpe = kpe
        desc_qpe = qpe
        desc_o = o
        desc_v = v

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    algo_map = {
        0: mla_decode_latent_sharedZ_ws_debug_txl,
    }

    algo_map[algo][grid](
        sm_scale, M,
        q.shape[0], q.shape[1], kv.shape[1],
        desc_q,
        desc_kv,
        desc_v,
        desc_o,
        q.shape[2], kv.shape[2],
        desc_qpe, desc_kpe,
        PE_DIM=HEAD_DIM_PE,
        R0=HEAD_DIM_Z,
        **extra_kern_args
    )

    return o

def ref_flash_attention_test(q, k, v, sm_scale, algo=0):
    HEAD_DIM_K = q.shape[-1]
    o = torch.empty_like(q)
    stage = 3
    extra_kern_args = {}
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    if supports_host_descriptor():
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    else:
        desc_q = q
        desc_k = k
        desc_v = v
        desc_o = o

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    
    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    algo_map = {
        0: _attn_fwd_ws_tma_txl1,
    }
    algo_map[algo][grid](
        sm_scale, M,
        q.shape[0], q.shape[1],
        desc_q, desc_k, desc_v, desc_o,
        q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        FP8_OUTPUT=False,
        STAGE=stage,
        warp_specialize=True,
        **extra_kern_args)
    return o

def ref_flash_attention(q, k, v, sm_scale):
    Z, H, N_Q, R0 = q.shape
    _, _, N_K, _ = k.shape

    o = torch.empty_like(q)
    m_all = torch.empty((Z, H, N_Q), device=q.device, dtype=torch.float32)

    for z in range(Z):
        for h in range(H):
            q_ = q[z, h]  # [N_Q, R0]
            k_ = k[z, h]  # [N_K, R0]
            v_ = v[z, h]  # [N_K, R0]

            qk = q_ @ k_.T  # [N_Q, N_K]
            qk = qk * sm_scale * 1.44269504  # log2(e)=1.44269504
            m_i = torch.max(qk, dim=1).values
            p = torch.pow(2, qk - m_i[:, None])
            l_i = torch.sum(p, dim=1)
            m_all[z, h] = m_i + torch.log2(l_i)
            o[z, h] = (p @ v_) / l_i[:, None]
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

def bench_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0, reps=1000, warmup_reps=1000):
    q = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    kv = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    qpe = (torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    kpe = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    sm_scale = 1 / math.sqrt(R0)

    bench_fn(
        f"mla Z{Z} H{H} NQ{N_Q} KH{KV_HEADS} KS{KV_SEQ_LEN} R0{R0} PE{PE_DIM} algo{algo}",
        reps,
        warmup_reps,
        lambda q, kv, qpe, kpe, sm_scale, algo: mla_test(q, kv, qpe, kpe, sm_scale, algo),
        q, kv, qpe, kpe, sm_scale, algo
    )    

def test_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0, no_tune=False):
    q = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    kv = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    qpe = (torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    kpe = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    # q = (torch.ones((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    # kv = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    # kvu = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN//2, R0), dtype=dtype, device=DEVICE)*0.5)
    # kvd = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN//2, R0), dtype=dtype, device=DEVICE)*2.0)
    # kv = torch.cat([kvu, kvd], dim=2)
    # qpe = (torch.ones((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    # qpe1 = (torch.randn((Z, H, N_Q-20, PE_DIM), dtype=dtype, device=DEVICE))
    # qpe2 = (torch.ones((Z, H, 20, PE_DIM), dtype=dtype, device=DEVICE))
    # qpe = torch.cat([qpe2, qpe1], dim=2)
    # kpe = (torch.ones((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    sm_scale = 1 / math.sqrt(R0)

    tri_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=algo)
    print(f"triton out: {tri_out}")
    # debug_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=5)
    ref_out = ref_mla(q, kv, qpe, kpe, sm_scale)

    max_err = (tri_out - ref_out).abs().max().item()
    location = (tri_out - ref_out).abs().argmax().item()
    # print(f"debug out: {debug_out}")
    print(f"ref out: {ref_out}")
    tri_out_around_location = tri_out.view(-1)[max(0, location-3):location+4]
    ref_out_around_location = ref_out.view(-1)[max(0, location-3):location+4]
    num_equal_locations = (tri_out - ref_out).abs() == max_err
    print(f"max err location indices: {torch.nonzero(num_equal_locations)}")
    print(f"tri around max err location: {tri_out_around_location}")
    print(f"ref around max err location: {ref_out_around_location}")
    print(f"Z{Z} H{H} NQ{N_Q} KH{KV_HEADS} KS{KV_SEQ_LEN} R0{R0} PE{PE_DIM} | max err: {max_err:.6f}")

def ref_op(Z, H, N_Q, R0, dtype=torch.float16, algo=0):
    q = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    k = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    v = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    sm_scale = 1 / math.sqrt(R0)

    tri_out = ref_flash_attention_test(q, k, v, sm_scale, algo=algo)
    ref_out = ref_flash_attention(q, k, v, sm_scale)

    max_err = (tri_out - ref_out).abs().max().item()
    print(f"Z{Z} H{H} NQ{N_Q} R0{R0} | max err: {max_err:.6f}")

def debug_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0):
    q = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    kv = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    v = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    qpe = (torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    kpe = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    sm_scale = 1 / math.sqrt(R0)

    tri_out = debug_test(q, kv, qpe, kpe, v, sm_scale, algo=algo)

def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    # if precision == 'fp8':
    #     metric_names = metric_names + ["tflop8/s"]
    # elif precision == 'fp16':
    #     metric_names = metric_names + ["tflop16/s"]
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)

if __name__ == "__main__":
    no_tune=True

    dump_dir="/workspace/dump/"
    print("TEST...")
    from triton import knobs

    # os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    knobs.runtime.override_arch='sm90'
    knobs.autotuning.print=True
    knobs.compilation.always_compile=True

    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir
        # knobs.compilation.override=True
        # knobs.cache.override_dir=dump_dir
    
    # debug_op(16, 32, 1024, 1, 1024, 256, 64, algo=0)
    # ref_op(16, 32, 1024, 256, algo=0)
    # test_op(16, 32, 1024, 1, 1024, 256, 64, algo=0, no_tune=no_tune)
    # test_op(16, 32, 1024, 1, 1024, 256, 64, algo=1, no_tune=no_tune)
    # test_op(1, 1, 64, 1, 128, 128, 64, algo=2, no_tune=no_tune)
    # test_op(1, 1, 256, 1, 256, 128, 64, algo=5, no_tune=no_tune)
    # test_op(16, 32, 1024, 1, 1024, 128, 64, algo=5, no_tune=no_tune)
    # test_op(16, 32, 1024, 1, 1024, 256, 64, algo=3, no_tune=no_tune)
    # test_op(16, 32, 1024, 1, 1024, 128, 64, algo=4, no_tune=no_tune)
    test_op(1, 1, 64, 1, 128, 128, 64, algo=7, no_tune=no_tune)
    # test_op(16, 32, 1024, 1, 1024, 512, 64, algo=7, no_tune=no_tune)
    # 如果blockn、blockm或者r0设置有一点高的化就会oom，得优化

    # proton.start("mla", hook="triton")
    # proton.deactivate()
    # bench_op(16, 32, 1024, 1, 1024, 512, 64, algo=6, reps=1000, warmup_reps=1000)
    # bench_op(16, 32, 1024, 1, 1024, 512, 64, algo=0, reps=1000, warmup_reps=1000)
    # # bench_op(16, 32, 1024, 1, 1024, 256, 64, algo=5, reps=1000, warmup_reps=1000)
    # # bench_op(16, 32, 1024, 1, 1024, 256, 64, algo=3, reps=1000, warmup_reps=1000)
    # proton.finalize()
    # show_profile("mla")