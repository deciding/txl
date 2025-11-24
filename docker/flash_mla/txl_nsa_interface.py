import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import txl

from typing import Optional, Tuple

import torch

GROUP_SIZE = tl.constexpr(8)
# NamedBarriers
wg0_bunch_0_ready = tl.constexpr(8)
wg1_bunch_0_ready = tl.constexpr(9)
wg0_s0_ready = tl.constexpr(10)
wg1_s1_ready = tl.constexpr(11)
sL_ready = tl.constexpr(12)
warpgroup0_sync = tl.constexpr(13)
warpgroup1_sync = tl.constexpr(14)
@txl.jit
#@txl.jit(src_file='dump/smem/SKCNG6F2XUQBA3XUKJ4ASFQM2E6HEJDVNB3J2MBCN23UCKHYIFJQ/txl_mla0.ptx')
#@txl.jit(diff_mode="llir", log_dir='dump/')
#@txl.jit(diff_mode="ttgir", diff_select=4, log_dir='dump/smem')
def txl_mla0(
        q_nope_desc, q_pe_desc,
        kv_nope_ptr, kv_pe_ptr,
        o_desc,
        max_logits_desc, lse_desc,
        indices_ptr,
        SCALE_LOG2: tl.constexpr,

        S_Q : tl.constexpr,
        S_KV : tl.constexpr,
        B_H : tl.constexpr,
        D_Q : tl.constexpr,
        D_V : tl.constexpr,
        NUM_HEAD_BLOCKS : tl.constexpr,
        TOPK : tl.constexpr,
        B_TOPK : tl.constexpr,
        STRIDE_KV_NOPE_0: tl.constexpr,
        STRIDE_KV_PE_0: tl.constexpr,
       ):

    tid = txl.tid(0)
    idx_in_warpgroup = tid % 128
    pid = tl.program_id(0)
    s_q_idx = pid // NUM_HEAD_BLOCKS
    q_h_idx = pid % NUM_HEAD_BLOCKS
    NUM_TOPK_BLOCKS: tl.constexpr = TOPK // B_TOPK


    offs_q = pid * B_H

    q_nope_buf = txl.smem_alloc([B_H, 512], dtype=tl.bfloat16, num_stages=1); sQnope = txl.get_buffer(q_nope_buf, 0)
    q_pe_buf = txl.smem_alloc([B_H, 64], dtype=tl.bfloat16, num_stages=1); sQpe = txl.get_buffer(q_pe_buf, 0)
    mbar_q_nope_buf = txl.mbar_alloc(1, num_stages=1); mbar_q_nope = txl.get_buffer(mbar_q_nope_buf, 0)
    mbar_q_pe_buf = txl.mbar_alloc(1, num_stages=1); mbar_q_pe = txl.get_buffer(mbar_q_pe_buf, 0)


    #index_layout1d: tl.constexpr = txl.BlockedLayout([8], [32], [4], [0])
    #index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    Knope_buf = txl.smem_alloc([B_TOPK, 512], dtype=tl.bfloat16, num_stages=2);
    Kpe_buf = txl.smem_alloc([B_TOPK, 64], dtype=tl.bfloat16, num_stages=2);
    sKnope0 = txl.get_buffer(Knope_buf, 0)
    sKpe0 = txl.get_buffer(Kpe_buf, 0)
    sKnope1 = txl.get_buffer(Knope_buf, 1)
    sKpe1 = txl.get_buffer(Kpe_buf, 1)

    sS0 = txl.get_buffer(Kpe_buf, 0) # reuse
    sS1_buf = txl.smem_alloc([B_H, B_TOPK], dtype=tl.bfloat16, num_stages=1);
    sS1 = txl.get_buffer(sS1_buf, 0)

    #sO_buf = txl.smem_alloc([B_H, 512], dtype=tl.bfloat16, num_stages=1);
    sO = txl.get_buffer(q_nope_buf, 0) # reuse Q

    is_kv_valid_layout0: tl.constexpr = txl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0]) # 8 threads are for dilation
    is_kv_valid_layout: tl.constexpr = txl.SliceLayout(dim=1, parent=is_kv_valid_layout0) # 8 threads are for dilation
    is_kv_valid_buf = txl.smem_alloc([B_TOPK], dtype=tl.int8, num_stages=2);
    is_kv_valid0 = txl.get_buffer(is_kv_valid_buf, 0)
    is_kv_valid1 = txl.get_buffer(is_kv_valid_buf, 1)

    layout_acc: tl.constexpr = txl.NVMMADistributedLayout([3, 0], [4, 1], [16, 64, 16])

    sM_buf = txl.smem_alloc([B_H], dtype=tl.float32, num_stages=1); sM = txl.get_buffer(sM_buf, 0)
    sL_buf = txl.smem_alloc([B_H], dtype=tl.float32, num_stages=2); sL0 = txl.get_buffer(sL_buf, 0); sL1 = txl.get_buffer(sL_buf, 1)
    layout_sM: tl.constexpr = txl.SliceLayout(dim=1, parent=layout_acc)
    layout_row: tl.constexpr = txl.SliceLayout(dim=0, parent=layout_acc)

    # mma barriers
    mbar_k0_free = txl.mbar_alloc(128, num_stages=2); mbar_k0_free0 = txl.get_buffer(mbar_k0_free, 0); mbar_k0_free1 = txl.get_buffer(mbar_k0_free, 1)
    mbar_k0_ready = txl.mbar_alloc(128, num_stages=2); mbar_k0_ready0 = txl.get_buffer(mbar_k0_ready, 0); mbar_k0_ready1 = txl.get_buffer(mbar_k0_ready, 1)
    mbar_k1_free = txl.mbar_alloc(128, num_stages=2); mbar_k1_free0 = txl.get_buffer(mbar_k1_free, 0); mbar_k1_free1 = txl.get_buffer(mbar_k1_free, 1)
    mbar_k1_ready = txl.mbar_alloc(128, num_stages=2); mbar_k1_ready0 = txl.get_buffer(mbar_k1_ready, 0); mbar_k1_ready1 = txl.get_buffer(mbar_k1_ready, 1)

    mbar_is_kv_valid_ready_buf = txl.mbar_alloc(16, num_stages=1); mbar_is_kv_valid_ready = txl.get_buffer(mbar_is_kv_valid_ready_buf, 0)


    if txl.is_warpgroup([0, 1]):
        txl.reg_alloc(216)
        if txl.is_warpgroup([0]):
            txl.mbar_expect(mbar_q_nope, B_H*512*2)
            txl.tma_load(sQnope, q_nope_desc, [offs_q, 0], mbar_q_nope)
            txl.mbar_expect(mbar_q_pe, B_H*64*2)
            txl.tma_load(sQpe, q_pe_desc, [offs_q, 0], mbar_q_pe)

        txl.mbar_wait(mbar_q_nope, 0)
        txl.mbar_wait(mbar_q_pe, 0)

        rP = tl.zeros([B_H, B_TOPK], dtype=tl.float32)
        rM = tl.zeros([B_H], dtype=tl.float32) - float("inf") # m_i
        rL = tl.zeros([B_H], dtype=tl.float32) # l_i
        rO = tl.zeros([B_H, 256], dtype=tl.float32) # D_V//2

        cur_bar_wait_phase = 0

        sKnope0l = txl.smem_slice(sKnope0, 0, 256, dim=1)
        # B_TOPK, 256
        sV0l = sKnope0l
        sKnope1l = txl.smem_slice(sKnope1, 0, 256, dim=1)
        # B_TOPK, 256
        sV1l = sKnope1l

        sKnope0r = txl.smem_slice(sKnope0, 256, 256, dim=1)
        # B_TOPK, 256
        sV0r = sKnope0r
        sKnope1r = txl.smem_slice(sKnope1, 256, 256, dim=1)
        # B_TOPK, 256
        sV1r = sKnope1r

        for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
            # iter (-1) rP0 = sQ @ sK0
            if txl.is_warpgroup([0]):
                if block_idx == 0:
                    # pipelined_wait_and_qkt_gemm_l
                    txl.mbar_wait(mbar_k0_ready0, cur_bar_wait_phase)
                    # slice 0-3
                    for tile_index in tl.static_range(0, 256, 64):
                        cur_sQ = txl.smem_slice(sQnope, tile_index, 64, dim=1)
                        cur_sK = txl.smem_slice(sKnope0, tile_index, 64, dim=1)
                        rP = tl.dot(cur_sQ, cur_sK.T, rP)

                    # pipelined_wait_and_qkt_gemm_r
                    txl.mbar_wait(mbar_k0_ready1, cur_bar_wait_phase)
                    # slice 4-7
                    for tile_index in tl.static_range(256, 512, 64):
                        cur_sQ = txl.smem_slice(sQnope, tile_index, 64, dim=1)
                        cur_sK = txl.smem_slice(sKnope0, tile_index, 64, dim=1)
                        rP = tl.dot(cur_sQ, cur_sK.T, rP)
                    rP = tl.dot(sQpe, sKpe0.T, rP) # [B_H, 64] x [64, B_TOPK]

                    txl.dot_wait(0)

            # rP1 = sQ @ sK1
            if txl.is_warpgroup([1]):
                # pipelined_wait_and_qkt_gemm_r
                txl.mbar_wait(mbar_k1_ready1, cur_bar_wait_phase)
                # slice 4-7
                for tile_index in tl.static_range(256, 512, 64):
                    cur_sQ = txl.smem_slice(sQnope, tile_index, 64, dim=1)
                    cur_sK = txl.smem_slice(sKnope1, tile_index, 64, dim=1)
                    rP = tl.dot(cur_sQ, cur_sK.T, rP)
                rP = tl.dot(sQpe, sKpe1.T, rP) # [B_H, 64] x [64, B_TOPK]

                # pipelined_wait_and_qkt_gemm_l
                txl.mbar_wait(mbar_k1_ready0, cur_bar_wait_phase)
                # slice 0-3
                for tile_index in tl.static_range(0, 256, 64):
                    cur_sQ = txl.smem_slice(sQnope, tile_index, 64, dim=1)
                    cur_sK = txl.smem_slice(sKnope1, tile_index, 64, dim=1)
                    rP = tl.dot(cur_sQ, cur_sK.T, rP)


                txl.dot_wait(0)

            # mask_rP
            txl.mbar_wait(mbar_is_kv_valid_ready, cur_bar_wait_phase)
            if txl.is_warpgroup([0]):
                #reg_is_kv_valid = txl.smem_load(is_kv_valid0, layout_acc) # mind the shape
                reg_is_kv_valid = txl.smem_load(is_kv_valid0, layout_row) # mind the shape
            else:
                #reg_is_kv_valid = txl.smem_load(is_kv_valid1, layout_acc) # mind the shape
                reg_is_kv_valid = txl.smem_load(is_kv_valid1, layout_row) # mind the shape
            rP = tl.where(reg_is_kv_valid != 0, rP, float('-inf'))

            if txl.is_warpgroup([1]):
                txl.bar_wait(wg0_bunch_0_ready, 256) # wait for wg0 to provide half of sM

            # online_softmax_and_rescale_o
            #txl.mbar_wait(mbar_is_kv_valid_ready, cur_bar_wait_phase)
            cur_max: tl.tensor = tl.max(rP, axis = 1) # reduce on 2 rows, and a warp reduce
            cur_max *= SCALE_LOG2
            # wg0: rM load from sM in the last round
            # wg1: rM load from sM written by wg0
            if txl.is_warpgroup([0]):
                new_maxs = tl.maximum(rM, cur_max) # m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            else:
                r_sM = txl.frag_smem_load(sM, [64], layout_sM)
                new_maxs = tl.maximum(r_sM, cur_max) # m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            scale_for_o = tl.exp2(rM - new_maxs) # alpha = tl.math.exp2(m_i - m_ij)
            rO *= scale_for_o[:, None] # broadcast, then mind the encoding

            rP = tl.exp2(rP * SCALE_LOG2 - new_maxs[:, None]) # qk = qk * qk_scale - m_ij[:, None]; p = tl.math.exp2(qk)
            # TODO: warp_reduce
            cur_sum = tl.sum(rP, axis=1) # l_ij = tl.sum(p, 1)
            rL = rL * scale_for_o + cur_sum # l_i = l_i * alpha + l_ij
            rS = rP.to(tl.bfloat16) # p = p.to(dtype)

            # wg0 save half, then wg1 save whole
            # TODO: syncwarp?
            if idx_in_warpgroup % 4 == 0: # only store for every 4 because they are the same
                #txl.frag_smem_store(sM, new_maxs, layout_sM)
                txl.smem_store(sM, new_maxs) # TODO: remove the layout of frag_smem_store

            rM = new_maxs # m_i = m_ij

            if txl.is_warpgroup([0]):
                # 1. NamedBarriers, half sM for wg1 is ready
                txl.bar_arrive(wg0_bunch_0_ready, 256) # inform wg1 that half sM is saved
                rO = tl.dot(rS, sV0l, rO) # O0 += S0 @ V0l, [B_H, B_TOPK] x [B_TOPK, 256], use scaled rO and ready rS
                txl.dot_wait(0)
                txl.mbar_arrive(mbar_k0_free0)

                # 1.1. NamedBarriers, wait full sM from wg1
                txl.bar_wait(wg1_bunch_0_ready, 256)
                new_rM = txl.frag_smem_load(sM, [64], layout_sM)
                scale_factors = tl.exp2(rM - new_rM)
                rM = new_rM

                # 1.2. scale_rS, wg0 need additional scale of new rM, rL and rO later
                rS = (rP * scale_factors[:, None]).to(tl.bfloat16)

                # 2. save_rS_to_sS
                txl.smem_store(sS0, rS) # check rS nvmma or not
                txl.bar_arrive(wg0_s0_ready, 256)

                # 3. wait for sS1
                txl.bar_wait(wg1_s1_ready, 256) # wait for wg1 s1

                # 3.1 rescale_rO, wg0 need additional scale of new rM, for rL and rO
                rO *= scale_factors[:, None] # final scale
                rL *= scale_factors # check layout

                # 3.2 wait for sS1
                rO = tl.dot(sS1, sV1l, rO) # O0 += S1 @ V1l, get the whole Vl

            if txl.is_warpgroup([1]):
                # 1. NamedBarriers, updated other half of sM for wg0 is ready
                txl.bar_arrive(wg1_bunch_0_ready, 256) # inform wg0 of whole sM and let it rescale
                rO = tl.dot(rS, sV1r, rO) # O1 += S1 @ V1r, [B_H, B_TOPK] x [B_TOPK, 256], should not release k1 mbar, bcoz no dot wait

                # 2. save_rS_to_sS, and use sS0
                txl.smem_store(sS1, rS) # check rS nvmma or not
                txl.bar_wait(wg0_s0_ready, 256) # get s0 from wg0
                rO = tl.dot(sS0, sV0r, rO) # O1 += S0 @ V0r
                txl.bar_arrive(wg1_s1_ready, 256) # tell wg0 s1 is ready

                # 3. wait all dots for s vr
                # sV1r
                txl.dot_wait(1)
                txl.mbar_arrive(mbar_k1_free1)
                # sV0r
                txl.dot_wait(0)
                txl.mbar_arrive(mbar_k0_free1)

            cur_bar_wait_phase ^= 1

            if txl.is_warpgroup([0]):
                if block_idx + 2 < NUM_TOPK_BLOCKS:
                    # pipelined_wait_and_qkt_gemm_l
                    txl.mbar_wait(mbar_k0_ready0, cur_bar_wait_phase)
                    # slice 0-3
                    for tile_index in tl.static_range(0, 256, 64):
                        cur_sQ = txl.smem_slice(sQnope, tile_index, 64, dim=1)
                        cur_sK = txl.smem_slice(sKnope0, tile_index, 64, dim=1)
                        rP = tl.dot(cur_sQ, cur_sK.T, rP)

                    txl.dot_wait(1)
                    # mark sV1l as free
                    txl.mbar_arrive(mbar_k1_free0)

                    # pipelined_wait_and_qkt_gemm_r
                    txl.mbar_wait(mbar_k0_ready1, cur_bar_wait_phase)
                    # slice 4-7
                    for tile_index in tl.static_range(256, 512, 64):
                        cur_sQ = txl.smem_slice(sQnope, tile_index, 64, dim=1)
                        cur_sK = txl.smem_slice(sKnope0, tile_index, 64, dim=1)
                        rP = tl.dot(cur_sQ, cur_sK.T, rP)
                    rP = tl.dot(sQpe, sKpe0.T, rP) # [B_H, 64] x [64, B_TOPK]

                    # The whole sQ @ sK0
                    txl.dot_wait(0)
                else:
                    txl.dot_wait(0)
                    # mark sV1l as free
                    txl.mbar_arrive(mbar_k1_free0)

        # After block_idx

        # reduce_L
        # TODO: reduce L on warp_reduce
        if txl.is_warpgroup([0]):
            if idx_in_warpgroup % 4 == 0: # only store for every 4 because they are the same
                #txl.frag_smem_store(sL0, rL, layout_sM)
                txl.smem_store(sL0, rL)
        if txl.is_warpgroup([1]):
            if idx_in_warpgroup % 4 == 0: # only store for every 4 because they are the same
                #txl.frag_smem_store(sL1, rL, layout_sM)
                txl.smem_store(sL1, rL)
        # all finished sL store
        txl.bar_wait(sL_ready, 256)
        if txl.is_warpgroup([0]):
            peer_L = txl.frag_smem_load(sL1, (64,), layout_sM)
        else:
            peer_L = txl.frag_smem_load(sL0, (64,), layout_sM)
        rL += peer_L

        # store_O
        scale_factors = tl.where(rL == 0.0, 1.0, 1.0/rL)
        cur_rO = (rO * scale_factors[:, None]).to(tl.bfloat16)
        #sO_index = 0
        #cur_sO = txl.smem_slice(sO, sO_index, 256, 1)
        #for tile_index in range(0, 256, 64):
        #    cur_tile_sO = txl.smem_slice((cur_sO, 0, 64, 1)
        #    txl.smem_store(cur_tile_sO, cur_rO) # TODO: split cur_rO
        #    txl.bar_wait(warpgroup0_sync, 128)
        #    txl.tma_store(cur_tile_sO, o_desc, [offs_q, tile_index])

        if txl.is_warpgroup([0]):
            cur_sO = txl.smem_slice(sO, 0, 256, 1)
            txl.smem_store(cur_sO, cur_rO)
            txl.bar_wait(warpgroup0_sync, 128)
            txl.tma_store(cur_sO, o_desc, [offs_q, 0])
        else:
            cur_sO = txl.smem_slice(sO, 256, 256, 1)
            txl.smem_store(cur_sO, cur_rO)
            txl.bar_wait(warpgroup1_sync, 128)
            txl.tma_store(cur_sO, o_desc, [offs_q, 256])

        #txl.bar_wait(warpgroup0_sync, 128)
        #o_desc.store(cur_rO, [offs_q, 0])

        if txl.is_warpgroup([1]):
            # save lse 
            final_max_logits = tl.where(rL == 0.0, float('-inf'), rM)
            final_lse = tl.where(rL == 0.0, float('-inf'), tl.log2(rL) + rM)
            max_logits_desc.store([offs_q], final_max_logits)
            lse_desc.store([offs_q], final_lse)


    if txl.is_warpgroup([2]):
        txl.reg_dealloc(72)

        gIndices = indices_ptr + s_q_idx * TOPK
        index_offs = tl.arange(0, B_TOPK * GROUP_SIZE) // GROUP_SIZE # 4 strided with stride 128//8, e.g. 0 16 32 48

        inc_index_offs = tl.reshape(tl.arange(0, B_TOPK * 64) % 64, (64, 64))
        inc_index_offs = txl.relayout(inc_index_offs, (64, 64), index_layout2d) # TODO: col 8x8, natually redundant or merged in reshape?

        cur_bar_wait_phase = 1

        #for block_idx in range(0, 2, 2):
        for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
            # buf_idx 0
            cur_index_nope_arr = ()
            cur_index_pe_arr = ()
            is_token_valid_arr = ()

            for buf_idx in tl.static_range(0, 2):
            #for buf_idx in tl.static_range(0, 1):
                cur_g_indices = gIndices + (block_idx + buf_idx) * B_TOPK

                cur_index = tl.load(cur_g_indices + index_offs) # 4 strided with stride 128//8
                cur_index_nope0 = cur_index * STRIDE_KV_NOPE_0 # 4 strided with 128
                cur_index_pe0 = cur_index * STRIDE_KV_PE_0
                cur_index_nope0 = tl.broadcast_to(cur_index_nope0[:, None], (512, 8)) # 4x8 strided with 128 rows
                cur_index_nope0 = tl.reshape(cur_index_nope0, (64, 64)) # make it col 8x8
                cur_index_nope0 = txl.relayout(cur_index_nope0, (64, 64), index_layout2d) # TODO: redundant?
                cur_index_nope0 = cur_index_nope0 + inc_index_offs
                cur_index_pe0 = tl.broadcast_to(cur_index_pe0[:, None], (512, 8))
                cur_index_pe0 = tl.reshape(cur_index_pe0, (64, 64))
                cur_index_pe0 = txl.relayout(cur_index_pe0, (64, 64), index_layout2d)
                cur_index_pe0 = cur_index_pe0 + inc_index_offs


                is_token_valid = (cur_index >=0) & (cur_index < S_KV)
                is_token_valid0 = tl.broadcast_to(is_token_valid[:, None], (512, 8)) # each thread repeat 8 times
                is_token_valid0 = tl.reshape(is_token_valid0, (64, 64)) # reshape auto makes it correct

                is_token_valid0 = txl.relayout(is_token_valid0, (64, 64), index_layout2d) #TODO: not working?

                cur_index_nope_arr = cur_index_nope_arr + (cur_index_nope0,)
                cur_index_pe_arr = cur_index_pe_arr + (cur_index_pe0,)
                is_token_valid_arr = is_token_valid_arr + (is_token_valid0,)



            # V0l
            txl.mbar_wait(mbar_k0_free0, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[0]
            #txl.async_load_wait(0)
            for tile_index in tl.static_range(0, 256, 64):
            #for tile_index in tl.static_range(0, 64, 64):
                cur_sKnope = txl.smem_slice(sKnope0, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[0] + tile_index
                txl.async_load(cur_sKnope, kv_nope_ptr+offs_kv, mask=is_token_valid, contiguity=8) # col 8x8
                #txl.async_load(cur_sKnope, kv_nope_ptr+offs_kv, contiguity=8)
                #txl.async_load_wait(0)
            txl.mbar_arrive(mbar_k0_ready0, track_async_op=True)

            # V1r
            txl.mbar_wait(mbar_k1_free1, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[1]
            for tile_index in tl.static_range(256, 512, 64):
            #for tile_index in tl.static_range(0, 64, 64):
                cur_sKnope = txl.smem_slice(sKnope1, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[1] + tile_index
                txl.async_load(cur_sKnope, kv_nope_ptr+offs_kv, mask=is_token_valid, contiguity=8)
                #txl.async_load(cur_sKnope, kv_nope_ptr+offs_kv, contiguity=8)
                #txl.async_load_wait(0)
            offs_kv = cur_index_pe_arr[1]
            txl.async_load(sKpe1, kv_pe_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            txl.mbar_arrive(mbar_k1_ready1, track_async_op=True)

            # V0r
            txl.mbar_wait(mbar_k0_free1, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[0]
            for tile_index in tl.static_range(256, 512, 64):
                cur_sKnope = txl.smem_slice(sKnope0, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[0] + tile_index
                txl.async_load(cur_sKnope, kv_nope_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            offs_kv = cur_index_pe_arr[0]
            txl.async_load(sKpe0, kv_pe_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            txl.mbar_arrive(mbar_k0_ready1, track_async_op=True)

            # V1l
            txl.mbar_wait(mbar_k1_free0, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[1]
            for tile_index in tl.static_range(0, 256, 64):
                cur_sKnope = txl.smem_slice(sKnope1, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[1] + tile_index
                txl.async_load(cur_sKnope, kv_nope_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            txl.mbar_arrive(mbar_k1_ready0, track_async_op=True)

            if tid % 8 == 0:
                is_token_valid0 = is_token_valid_arr[0]
                is_token_valid1 = is_token_valid_arr[1]
                # smem and layout are same shape
                txl.frag_smem_store(is_kv_valid0, is_token_valid0.to(tl.int8), is_kv_valid_layout) # frag: only the first for each thread
                txl.frag_smem_store(is_kv_valid1, is_token_valid1.to(tl.int8), is_kv_valid_layout)
                txl.mbar_arrive(mbar_is_kv_valid_ready)

            cur_bar_wait_phase ^= 1

def txl_mla(
        #q: torch.Tensor,
        #kv: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_nope: torch.Tensor,
        kv_pe: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
    ):
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    #dump_dir='dump/smem/'
    dump_dir = None

    from triton import knobs
    import os
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-legalize"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    knobs.runtime.override_arch='sm90'
    #knobs.autotuning.print=True
    #knobs.compilation.always_compile=True

    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir


    B_H = 64
    B_TOPK = 64
    D_Q = 576
    D_K = D_Q
    D_V = 512

    s_q = q_nope.size(0)
    s_kv = kv_nope.size(0)
    top_k = indices.size(2)
    h_q = q_nope.size(1)

    h_kv = kv_nope.size(1)
    assert h_kv == 1
    d_qk = D_Q
    d_v = D_V

    qk_scale = sm_scale * 1.44269504

    #q_nope, q_pe = torch.split(q, [512, 64], dim=-1)
    #q_nope = q_nope.contiguous()
    #q_pe = q_pe.contiguous()

    #kv_nope, kv_pe = torch.split(kv, [512, 64], dim=-1)
    #kv_nope = kv_nope.contiguous()
    #kv_pe = kv_pe.contiguous()


    out = torch.empty((s_q, h_q, d_v), dtype=q_nope.dtype, device=q_nope.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q_nope.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q_nope.device)

    q_nope_desc = TensorDescriptor(q_nope, (s_q*h_q, 512), (512, 1), [B_H, 512])
    q_pe_desc = TensorDescriptor(q_pe, (s_q*h_q, 64), (64, 1), [B_H, 64])
    o_desc = TensorDescriptor(out, (s_q*h_q, d_v), (d_v, 1), [B_H, D_V])
    max_logits_desc = TensorDescriptor(max_logits, (s_q*h_q, ), (1, ), [B_H])
    lse_desc = TensorDescriptor(lse, (s_q*h_q, ), (1, ), [B_H])

    # TESTS
    #out1 = torch.empty((s_q, h_q, d_v//2), dtype=q.dtype, device=q.device)
    #out2 = torch.empty((s_q, h_q, d_v//2), dtype=q.dtype, device=q.device)
    #o1_desc = TensorDescriptor(out1, (s_q*h_q, d_v//2), (d_v//2, 1), [B_H, D_V//2])
    #o2_desc = TensorDescriptor(out2, (s_q*h_q, d_v//2), (d_v//2, 1), [B_H, D_V//2])

    NUM_HEAD_BLOCKS = h_q // B_H
    txl_mla0[(NUM_HEAD_BLOCKS * s_q,)](
            q_nope_desc, q_pe_desc,
            kv_nope, kv_pe,
            o_desc,
            max_logits_desc, lse_desc,
            indices,

            qk_scale,
            s_q, s_kv,
            B_H, D_Q, D_V,
            NUM_HEAD_BLOCKS,
            top_k,
            B_TOPK,
            kv_nope.stride(0),
            kv_pe.stride(0),
            num_warps=4, num_warpgroups=3)
    return out, max_logits, lse

def make_txl_mla_runner(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_pe: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    dump_dir=None,
):
    
    from triton import knobs

    knobs.runtime.override_arch = "sm90"
    # knobs.autotuning.print = True
    # knobs.compilation.always_compile = True

    if dump_dir is not None:
        knobs.compilation.dump_ir = True
        knobs.cache.dump_dir = dump_dir

    B_H = 64
    B_TOPK = 64
    D_Q = 576
    D_K = D_Q
    D_V = d_v 

    s_q = q_nope.size(0)
    s_kv = kv_nope.size(0)
    top_k = indices.size(2)
    h_q = q_nope.size(1)
    h_kv = kv_nope.size(1)
    assert h_kv == 1

    d_qk = D_Q
    d_v = D_V

    qk_scale = sm_scale * 1.44269504

    out = torch.empty((s_q, h_q, d_v), dtype=q_nope.dtype, device=q_nope.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q_nope.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q_nope.device)

    q_nope_desc = TensorDescriptor(q_nope, (s_q * h_q, 512), (512, 1), [B_H, 512])
    q_pe_desc   = TensorDescriptor(q_pe,   (s_q * h_q, 64),  (64, 1),  [B_H, 64])
    o_desc      = TensorDescriptor(out,    (s_q * h_q, d_v), (d_v, 1), [B_H, D_V])
    max_logits_desc = TensorDescriptor(max_logits, (s_q * h_q,), (1,), [B_H])
    lse_desc        = TensorDescriptor(lse,        (s_q * h_q,), (1,), [B_H])

    NUM_HEAD_BLOCKS = h_q // B_H

    grid = (NUM_HEAD_BLOCKS * s_q,) 

    def runner():
        txl_mla0[grid](
            q_nope_desc, q_pe_desc,
            kv_nope, kv_pe,
            o_desc,
            max_logits_desc, lse_desc,
            indices,
            qk_scale,
            s_q, s_kv,
            B_H, D_Q, D_V,
            NUM_HEAD_BLOCKS,
            top_k,
            B_TOPK,
            kv_nope.stride(0),
            kv_pe.stride(0),
            num_warps=4,
            num_warpgroups=3,
        )
        return out, max_logits, lse

    return runner
