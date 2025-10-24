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

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

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

ws_dim_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)

@txl.autotune(
    configs=[txl.Config(ws_dim_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook)],
    key=["KV_SEQ_LEN","R0","PE_DIM"]
)
@txl.jit
def mla_txl_kernel( # tilelange ws 
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

        # mma_layout: tl.constexpr = txl.NVMMADistributedLayout(
        #     version=[3, 0],           
        #     warps_per_cta=[4, 1],     
        #     instr_shape=[16, 64, 16], 
        # )

        # alpha_reg_layout: tl.constexpr = txl.SliceLayout(
        #     dim=1,            
        #     parent=mma_layout
        # )

        for _n in range(0, KV_SEQ_LEN, BLOCK_N):
            cur_mZ = txl.get_buffer(mZ, bufIdxR)
            txl.mbar_wait(cur_mZ, phase)
            cur_Z = txl.get_buffer(bZ, bufIdxR)
            cur_ZR = txl.smem_slice(cur_Z, R0//2, R0//2, 1)
            
            ##### bug place #####
            txl.mbar_wait(txl.get_buffer(bar_SS_ready, bufIdxR), phase)

            p = txl.get_buffer(S_shared, bufIdxR)
            alpha = txl.get_buffer(ALPHA_sh, bufIdxR)

            # alpha_reg = txl.smem_load(alpha, alpha_reg_layout)
            # accR = accR * alpha_reg[:, None]
            accR = accR * alpha[:, None]
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
        0: mla_txl_kernel,
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

def test_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0, no_tune=False):
    q = (torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE))
    kv = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE))
    qpe = (torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE))
    kpe = (torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE))
    sm_scale = 1 / math.sqrt(R0)

    tri_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=algo)
    print(f"triton out: {tri_out}")
    # debug_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=5)
    ref_out = ref_mla(q, kv, qpe, kpe, sm_scale)

    max_err = (tri_out - ref_out).abs().max().item()
    # print(f"debug out: {debug_out}")
    print(f"ref out: {ref_out}")
    print(f"Z{Z} H{H} NQ{N_Q} KH{KV_HEADS} KS{KV_SEQ_LEN} R0{R0} PE{PE_DIM} | max err: {max_err:.6f}")

if __name__ == "__main__":
    no_tune=True

    print("TEST...")

    test_op(16, 32, 1024, 1, 1024, 256, 64, algo=0, no_tune=no_tune)