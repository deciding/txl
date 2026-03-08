import triton
import triton.language as tl
import teraxlang as txl
import torch
import os
import sys
import math
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

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == 'cuda'

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9
tma_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)

def _pre_hook(nargs):
    BM = nargs['BLOCK_M']
    BN = nargs['BLOCK_N']
    D = nargs['D']
    PE = nargs['PE_DIM']
    if not isinstance(nargs['desc_qhat'], TensorDescriptor):
        return
    nargs['desc_qhat'].block_shape = [BM, D]
    nargs['desc_qpe'].block_shape = [BM, PE]
    nargs['desc_zkv'].block_shape = [BN, D // 2]
    nargs['desc_kpe'].block_shape = [BN, PE]
    nargs['desc_o_lat'].block_shape = [BM, D // 2]
ws_cuta_cfg = dict(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=1)

@txl.autotune(configs=[txl.Config(ws_cuta_cfg, num_stages=1, num_warps=4, num_warpgroups=3, pre_hook=_pre_hook)], key=['KV_SEQ_LEN', 'D', 'PE_DIM'])
@txl.jit
def mla_txl(sm_scale, M, B, H, H_KV, desc_qhat, desc_zkv, desc_o_lat, N_Q, KV_SEQ_LEN, desc_qpe, desc_kpe, PE_DIM: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr, NUM_SMS: tl.constexpr):
    tl.static_assert(BLOCK_N <= D)
    tl.static_assert(D % 2 == 0)
    tl.static_assert(PE_DIM == BLOCK_N)
    tl.static_assert(NUM_STAGES == 1)
    dtype = tl.float16
    tidx = txl.tid(0)
    pid_m = tl.program_id(0)
    off_kvh_SMs = tl.program_id(1)
    batch_per_cta = B // NUM_SMS
    off_kvh = off_kvh_SMs // NUM_SMS
    off_SMs = off_kvh_SMs % NUM_SMS
    begin_batch = off_SMs * batch_per_cta
    end_batch = begin_batch + batch_per_cta
    rows_q = B * H * N_Q
    rows_kv = B * H_KV * KV_SEQ_LEN
    desc_qhat = _maybe_desc(desc_qhat, [rows_q, D], [D, 1], [BLOCK_M, D])
    desc_o_lat = _maybe_desc(desc_o_lat, [rows_q, D], [D, 1], [BLOCK_M, D])
    desc_zkv = _maybe_desc(desc_zkv, [rows_kv, D], [D, 1], [BLOCK_N, D])
    desc_qpe = _maybe_desc(desc_qpe, [rows_q, PE_DIM], [PE_DIM, 1], [BLOCK_M, PE_DIM])
    desc_kpe = _maybe_desc(desc_kpe, [rows_kv, PE_DIM], [PE_DIM, 1], [BLOCK_N, PE_DIM])
    bQ = txl.smem_alloc([BLOCK_M, D], dtype=dtype)
    bQpe = txl.smem_alloc([BLOCK_M, PE_DIM], dtype=dtype)
    mQ = txl.mbar_alloc(1)
    mQpe = txl.mbar_alloc(1)
    bZL0 = txl.smem_alloc([BLOCK_N, D // 2], dtype=dtype, num_stages=1)
    bZR0 = txl.smem_alloc([BLOCK_N, D // 2], dtype=dtype, num_stages=1)
    bZL1 = txl.smem_alloc([BLOCK_N, D // 2], dtype=dtype, num_stages=1)
    bZR1 = txl.smem_alloc([BLOCK_N, D // 2], dtype=dtype, num_stages=1)
    bKpe0 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=1)
    bKpe1 = txl.smem_alloc([BLOCK_N, PE_DIM], dtype=dtype, num_stages=1)
    mZL0 = txl.mbar_alloc(1, num_stages=1)
    mZR0 = txl.mbar_alloc(1, num_stages=1)
    mZL1 = txl.mbar_alloc(1, num_stages=1)
    mZR1 = txl.mbar_alloc(1, num_stages=1)
    mKpe0 = txl.mbar_alloc(1, num_stages=1)
    mKpe1 = txl.mbar_alloc(1, num_stages=1)
    mQK0 = txl.mbar_alloc(128, num_stages=1)
    mQK1 = txl.mbar_alloc(128, num_stages=1)
    mPV0_L = txl.mbar_alloc(128, num_stages=1)
    mPV0_R = txl.mbar_alloc(128, num_stages=1)
    mPV1_L = txl.mbar_alloc(128, num_stages=1)
    mPV1_R = txl.mbar_alloc(128, num_stages=1)
    mQ0 = txl.mbar_alloc(128, num_stages=1)
    mQ1 = txl.mbar_alloc(128, num_stages=1)
    bP0 = txl.smem_alloc([BLOCK_M, BLOCK_N], dtype=dtype, num_stages=1)
    bP1 = bQpe
    bMax = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=1)
    bL0 = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=1)
    bL1 = txl.smem_alloc([BLOCK_M], dtype=tl.float32, num_stages=1)
    mma_layout: tl.constexpr = txl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16])
    max_reg_layout: tl.constexpr = txl.SliceLayout(dim=1, parent=mma_layout)
    A_op_layout: tl.constexpr = txl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    cur_bQ = txl.get_buffer(bQ, 0)
    cur_mQ = txl.get_buffer(mQ, 0)
    cur_bQpe = txl.get_buffer(bQpe, 0)
    cur_mQpe = txl.get_buffer(mQpe, 0)
    cur_mZL0 = txl.get_buffer(mZL0, 0)
    cur_mZR0 = txl.get_buffer(mZR0, 0)
    cur_bZL0 = txl.get_buffer(bZL0, 0)
    cur_bZR0 = txl.get_buffer(bZR0, 0)
    cur_bKpe0 = txl.get_buffer(bKpe0, 0)
    cur_mKpe0 = txl.get_buffer(mKpe0, 0)
    cur_mZL1 = txl.get_buffer(mZL1, 0)
    cur_mZR1 = txl.get_buffer(mZR1, 0)
    cur_bZL1 = txl.get_buffer(bZL1, 0)
    cur_bZR1 = txl.get_buffer(bZR1, 0)
    cur_bKpe1 = txl.get_buffer(bKpe1, 0)
    cur_mKpe1 = txl.get_buffer(mKpe1, 0)
    cur_mQK0 = txl.get_buffer(mQK0, 0)
    cur_mQK1 = txl.get_buffer(mQK1, 0)
    cur_mPV0_L = txl.get_buffer(mPV0_L, 0)
    cur_mPV0_R = txl.get_buffer(mPV0_R, 0)
    cur_mPV1_L = txl.get_buffer(mPV1_L, 0)
    cur_mPV1_R = txl.get_buffer(mPV1_R, 0)
    cur_mQ0 = txl.get_buffer(mQ0, 0)
    cur_mQ1 = txl.get_buffer(mQ1, 0)
    cur_Max = txl.get_buffer(bMax, 0)
    cur_P0 = txl.get_buffer(bP0, 0)
    cur_P1 = txl.get_buffer(bP1, 0)
    cur_L0 = txl.get_buffer(bL0, 0)
    cur_L1 = txl.get_buffer(bL1, 0)
    if txl.is_warpgroup([0]):
        txl.reg_dealloc(40)
        phase = 1
        q_phase = 0
        for off_z in range(begin_batch, end_batch):
            heads_per_kv = H // H_KV
            q_base = off_z * (H * N_Q) + off_kvh * (heads_per_kv * N_Q)
            qo_off = q_base + pid_m * BLOCK_M
            kv_head_idx = off_kvh
            kv_base = off_z * (H_KV * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
            kv_off = kv_base
            txl.mbar_expect(cur_mQ, BLOCK_M * D * 2)
            txl.mbar_expect(cur_mQpe, BLOCK_M * PE_DIM * 2)
            txl.mbar_wait(cur_mQ0, q_phase ^ 1)
            txl.mbar_wait(cur_mQ1, q_phase ^ 1)
            txl.tma_load(cur_bQ, desc_qhat, [qo_off, 0], cur_mQ)
            txl.mbar_wait(cur_mQ, q_phase)
            txl.tma_load(cur_bQpe, desc_qpe, [qo_off, 0], cur_mQpe)
            txl.mbar_wait(cur_mQpe, q_phase)
            q_phase ^= 1
            for _n in range(0, KV_SEQ_LEN, BLOCK_N * 2):
                txl.mbar_wait(cur_mQK0, phase)
                txl.mbar_expect(cur_mKpe0, BLOCK_N * PE_DIM * 2)
                txl.tma_load(cur_bKpe0, desc_kpe, [kv_off, 0], cur_mKpe0)
                txl.mbar_wait(cur_mQK1, phase)
                txl.mbar_expect(cur_mKpe1, BLOCK_N * PE_DIM * 2)
                txl.tma_load(cur_bKpe1, desc_kpe, [kv_off + BLOCK_N, 0], cur_mKpe1)
                txl.mbar_wait(cur_mPV0_L, phase)
                txl.mbar_expect(cur_mZL0, BLOCK_N * D // 2 * 2)
                txl.tma_load(cur_bZL0, desc_zkv, [kv_off, 0], cur_mZL0)
                txl.mbar_wait(cur_mPV1_R, phase)
                txl.mbar_expect(cur_mZR1, BLOCK_N * D // 2 * 2)
                txl.tma_load(cur_bZR1, desc_zkv, [kv_off + BLOCK_N, D // 2], cur_mZR1)
                txl.mbar_wait(cur_mPV0_R, phase)
                txl.mbar_expect(cur_mZR0, BLOCK_N * D // 2 * 2)
                txl.tma_load(cur_bZR0, desc_zkv, [kv_off, D // 2], cur_mZR0)
                txl.mbar_wait(cur_mPV1_L, phase)
                txl.mbar_expect(cur_mZL1, BLOCK_N * D // 2 * 2)
                txl.tma_load(cur_bZL1, desc_zkv, [kv_off + BLOCK_N, 0], cur_mZL1)
                kv_off += 2 * BLOCK_N
                phase ^= 1
    if txl.is_warpgroup([1]):
        txl.reg_alloc(232)
        phase = 0
        q_phase = 0
        for off_z in range(begin_batch, end_batch):
            heads_per_kv = H // H_KV
            q_base = off_z * (H * N_Q) + off_kvh * (heads_per_kv * N_Q)
            qo_off = q_base + pid_m * BLOCK_M
            kv_head_idx = off_kvh
            kv_base = off_z * (H_KV * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
            kv_off = kv_base
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            qk_scale = sm_scale * 1.44269504
            m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
            l_i = tl.full([BLOCK_M], 1.0, tl.float32)
            accL = tl.zeros([BLOCK_M, D // 2], dtype=tl.float32)
            txl.mbar_wait(cur_mQ, q_phase)
            txl.mbar_wait(cur_mQpe, q_phase)
            rQpe = txl.smem_load(cur_bQpe)
            cur_bQl = txl.smem_slice(cur_bQ, 0, D // 2, 1)
            cur_bQr = txl.smem_slice(cur_bQ, D // 2, D // 2, 1)
            q_phase ^= 1
            for _n in range(0, KV_SEQ_LEN, BLOCK_N * 2):
                txl.mbar_wait(cur_mKpe0, phase)
                acc_s = tl.dot(rQpe, cur_bKpe0.T)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mQK0)
                txl.mbar_wait(cur_mZL0, phase)
                acc_s += tl.dot(cur_bQl, cur_bZL0.T)
                txl.dot_wait(0)
                txl.mbar_wait(cur_mZR0, phase)
                acc_s += tl.dot(cur_bQr, cur_bZR0.T)
                txl.dot_wait(0)
                m_ij0 = tl.maximum(m_i, tl.max(acc_s, 1) * qk_scale)
                alpha0 = tl.math.exp2(m_i - m_ij0)
                txl.smem_store(cur_Max, m_ij0)
                txl.bar_arrive(12, 256)
                acc_s = acc_s * qk_scale - m_ij0[:, None]
                p0 = tl.math.exp2(acc_s)
                l_ij0 = tl.sum(p0, 1)
                l_i = l_i * alpha0 + l_ij0
                accL = accL * alpha0[:, None]
                m_i = m_ij0
                accL = tl.dot(p0.to(dtype), cur_bZL0, accL)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mPV0_L)
                txl.bar_wait(14, 256)
                m_ij1 = txl.smem_load(cur_Max)
                alpha1 = tl.math.exp2(m_i - m_ij1)
                m_i = m_ij1
                p0 = p0 * alpha1[:, None]
                txl.smem_store(cur_P0, p0.to(dtype))
                txl.bar_arrive(11, 256)
                txl.mbar_wait(cur_mZL1, phase)
                accL = accL * alpha1[:, None]
                l_i = l_i * alpha1
                txl.bar_wait(10, 256)
                txl.fence_proxy_async()
                accL = tl.dot(cur_P1.to(dtype), cur_bZL1, accL)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mPV1_L)
                phase ^= 1
            txl.smem_store(cur_L0, l_i)
            txl.bar_arrive(9, 256)
            txl.bar_wait(8, 256)
            L1_reg = txl.smem_load(cur_L1)
            l_i = l_i + L1_reg
            m_i += tl.math.log2(l_i)
            accL = accL / l_i[:, None]
            txl.smem_store(cur_bZL0, accL.to(dtype))
            txl.tma_store(cur_bZL0, desc_o_lat, [qo_off, 0])
            txl.mbar_arrive(cur_mQ0)
    if txl.is_warpgroup([2]):
        txl.reg_alloc(232)
        phase = 0
        q_phase = 0
        for off_z in range(begin_batch, end_batch):
            heads_per_kv = H // H_KV
            q_base = off_z * (H * N_Q) + off_kvh * (heads_per_kv * N_Q)
            qo_off = q_base + pid_m * BLOCK_M
            kv_head_idx = off_kvh
            kv_base = off_z * (H_KV * KV_SEQ_LEN) + kv_head_idx * KV_SEQ_LEN
            kv_off = kv_base
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            qk_scale = sm_scale * 1.44269504
            m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
            l_i = tl.full([BLOCK_M], 1.0, tl.float32)
            accR = tl.zeros([BLOCK_M, D // 2], dtype=tl.float32)
            txl.mbar_wait(cur_mQ, q_phase)
            txl.mbar_wait(cur_mQpe, q_phase)
            rQpe = txl.smem_load(cur_bQpe)
            cur_bQl = txl.smem_slice(cur_bQ, 0, D // 2, 1)
            cur_bQr = txl.smem_slice(cur_bQ, D // 2, D // 2, 1)
            q_phase ^= 1
            for _n in range(0, KV_SEQ_LEN, BLOCK_N * 2):
                txl.mbar_wait(cur_mKpe1, phase)
                acc_s = tl.dot(rQpe, cur_bKpe1.T)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mQK1)
                txl.mbar_wait(cur_mZR1, phase)
                acc_s += tl.dot(cur_bQr, cur_bZR1.T)
                txl.dot_wait(0)
                txl.mbar_wait(cur_mZL1, phase)
                acc_s += tl.dot(cur_bQl, cur_bZL1.T)
                txl.dot_wait(0)
                txl.bar_wait(12, 256)
                m_ij0 = txl.smem_load(cur_Max)
                m_ij1 = tl.maximum(m_ij0, tl.max(acc_s, 1) * qk_scale)
                alpha1 = tl.math.exp2(m_i - m_ij1)
                acc_s = acc_s * qk_scale - m_ij1[:, None]
                p1 = tl.math.exp2(acc_s)
                l_ij1 = tl.sum(p1, 1)
                l_i = l_i * alpha1 + l_ij1
                accR = accR * alpha1[:, None]
                m_i = m_ij1
                txl.smem_store(cur_Max, m_ij1)
                txl.bar_arrive(14, 256)
                accR = tl.dot(p1.to(dtype), cur_bZR1, accR)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mPV1_R)
                txl.bar_wait(11, 256)
                txl.mbar_wait(cur_mZR0, phase)
                txl.fence_proxy_async()
                accR = tl.dot(cur_P0.to(dtype), cur_bZR0, accR)
                txl.dot_wait(0)
                txl.mbar_arrive(cur_mPV0_R)
                txl.smem_store(cur_P1, p1.to(dtype))
                txl.bar_arrive(10, 256)
                phase ^= 1
            txl.smem_store(cur_L1, l_i)
            txl.bar_arrive(8, 256)
            txl.bar_wait(9, 256)
            L0_reg = txl.smem_load(cur_L0)
            l_i = l_i + L0_reg
            accR = accR / l_i[:, None]
            txl.smem_store(cur_bZL1, accR.to(dtype))
            txl.tma_store(cur_bZL1, desc_o_lat, [qo_off, D // 2])
            txl.mbar_arrive(cur_mQ1)

def mla_test(q, kv, qpe, kpe, sm_scale, algo=0):
    HEAD_DIM_Q = q.shape[-1]
    HEAD_DIM_Z = kv.shape[-1]
    HEAD_DIM_PE = qpe.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_Z
    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    o = torch.empty_like(q)
    extra_kern_args = {}
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    if supports_host_descriptor():
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
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)
    q_heads_per_kv_heads = q.shape[1] // kv.shape[1]
    total_q_seqlen = q_heads_per_kv_heads * q.shape[2]

    def grid(META):
        return (triton.cdiv(total_q_seqlen, META['BLOCK_M']), kv.shape[1] * NUM_SMS, 1)
    algo_map = {0: mla_txl}
    algo_map[algo][grid](sm_scale, M, q.shape[0], q.shape[1], kv.shape[1], desc_q, desc_kv, desc_o, q.shape[2], kv.shape[2], desc_qpe, desc_kpe, PE_DIM=HEAD_DIM_PE, D=HEAD_DIM_Z, NUM_SMS=NUM_SMS, **extra_kern_args)
    return o

def make_mla_runner_txl(q, kv, qpe, kpe, sm_scale, algo=0):
    HEAD_DIM_Q = q.shape[-1]
    HEAD_DIM_Z = kv.shape[-1]
    HEAD_DIM_PE = qpe.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_Z
    NUM_SMS = 132
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    if supports_host_descriptor():
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]
        kv_dim = kv.shape[0] * kv.shape[1] * kv.shape[2]
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
        desc_kv = TensorDescriptor(kv, shape=[kv_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
        desc_qpe = TensorDescriptor(qpe, shape=[y_dim, HEAD_DIM_PE], strides=[HEAD_DIM_PE, 1], block_shape=dummy_block)
        desc_kpe = TensorDescriptor(kpe, shape=[kv_dim, HEAD_DIM_PE], strides=[HEAD_DIM_PE, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_Z], strides=[HEAD_DIM_Z, 1], block_shape=dummy_block)
    else:
        desc_q, desc_kv, desc_qpe, desc_kpe, desc_o = (q, kv, qpe, kpe, o)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)
    q_heads_per_kv_heads = q.shape[1] // kv.shape[1]
    total_q_seqlen = q_heads_per_kv_heads * q.shape[2]

    def grid(META):
        return (triton.cdiv(total_q_seqlen, META['BLOCK_M']), kv.shape[1] * NUM_SMS, 1)
    algo_map = {0: mla_txl}
    kern = algo_map[algo]

    def run_once():
        kern[grid](sm_scale, M, q.shape[0], q.shape[1], kv.shape[1], desc_q, desc_kv, desc_o, q.shape[2], kv.shape[2], desc_qpe, desc_kpe, PE_DIM=HEAD_DIM_PE, D=HEAD_DIM_Z, NUM_SMS=NUM_SMS)
        return o
    return run_once

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
            q_ = q[z, h]
            qpe_ = qpe[z, h]
            kv_ = kv[z, kv_head_idx]
            kpe_ = kpe[z, kv_head_idx]
            qk = q_ @ kv_.T + qpe_ @ kpe_.T
            qk = qk * sm_scale * 1.44269504
            m_i = torch.max(qk, dim=1).values
            p = torch.pow(2, qk - m_i[:, None])
            l_i = torch.sum(p, dim=1)
            m_all[z, h] = m_i + torch.log2(l_i)
            o[z, h] = p @ kv_ / l_i[:, None]
    return o

@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)

def bench_fn(label, reps, warmup_reps, fn, *args):
    print(f'Benchmarking {label}: ...', end='')
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)
    print(f'\rBenchmarking {label}: done')

def bench_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0, reps=1000, warmup_reps=1000):
    q = torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE)
    kv = torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE)
    qpe = torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE)
    kpe = torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE)
    sm_scale = 1 / math.sqrt(R0)
    bench_fn(f'mla Z{Z} H{H} NQ{N_Q} KH{KV_HEADS} KS{KV_SEQ_LEN} R0{R0} PE{PE_DIM} algo{algo}', reps, warmup_reps, lambda q, kv, qpe, kpe, sm_scale, algo: mla_test(q, kv, qpe, kpe, sm_scale, algo), q, kv, qpe, kpe, sm_scale, algo)

def test_op(Z, H, N_Q, KV_HEADS, KV_SEQ_LEN, R0, PE_DIM, dtype=torch.float16, algo=0, no_tune=False):
    q = torch.randn((Z, H, N_Q, R0), dtype=dtype, device=DEVICE)
    kv = torch.randn((Z, KV_HEADS, KV_SEQ_LEN, R0), dtype=dtype, device=DEVICE)
    qpe = torch.randn((Z, H, N_Q, PE_DIM), dtype=dtype, device=DEVICE)
    kpe = torch.randn((Z, KV_HEADS, KV_SEQ_LEN, PE_DIM), dtype=dtype, device=DEVICE)
    sm_scale = 1 / math.sqrt(R0)
    tri_out = mla_test(q, kv, qpe, kpe, sm_scale, algo=algo)
    print('finish')
    print(tri_out.shape)
    print(f'triton out: {tri_out[:1, :1, :5, :5]}')
    ref_out = ref_mla(q, kv, qpe, kpe, sm_scale)
    max_err = (tri_out - ref_out).abs().max().item()
    print(f'Z{Z} H{H} NQ{N_Q} KH{KV_HEADS} KS{KV_SEQ_LEN} R0{R0} PE{PE_DIM} | max err: {max_err:.6f}')

def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ['time/ms']
    file_name = f'{profile_name}.hatchet'
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)
if __name__ == '__main__':
    no_tune = True
    dump_dir = None
    print('TEST...')
    from triton import knobs
    knobs.autotuning.print = True
    knobs.compilation.always_compile = True
    if dump_dir:
        knobs.compilation.dump_ir = True
        knobs.cache.dump_dir = dump_dir
    B = 114
    test_op(B, 32, 64, 1, 256, 512, 64, algo=0, no_tune=no_tune)
    proton.start('mla', hook='triton')
    proton.deactivate()
    bench_op(B, 32, 64, 1, 256, 512, 64, algo=0, reps=100, warmup_reps=100)
    proton.finalize()
    show_profile('mla')
