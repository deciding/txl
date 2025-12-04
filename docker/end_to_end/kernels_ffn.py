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

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret

def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    NUM_CONSUMER_GROUPS = nargs.get("NUM_CONSUMER_GROUPS", 1)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_M //= NUM_CONSUMER_GROUPS
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]

@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
                "NUM_STAGES": 2,
            },
            num_stages=2,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)

@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_ws_tma_txl_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,

    # 3.4.x
    #EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #

    # fnn
    apply_gelu: tl.constexpr=False,
    bias_ptr=None,
    residual_ptr=None,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    byte_count: tl.constexpr = 2 if dtype == tl.float16 else 1
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    a0 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    a1 = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b0 = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer_a0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_a1 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_producer_b0 = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    mbar_consumer1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
    mbar_consumer2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)


    if txl.is_warpgroup([0]):

        phase = 1
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)

                mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                a0_buf = txl.get_buffer(a0, bufIdx)
                a1_buf = txl.get_buffer(a1, bufIdx)
                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_c1, phase)
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M//2*BLOCK_SIZE_K*byte_count)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)

                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N*BLOCK_SIZE_K*byte_count)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)


                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M//2*BLOCK_SIZE_K*byte_count)
                txl.tma_load(a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0:
                    phase = phase^1

    if txl.is_warpgroup([1, 2]): # TODO: else
        phase = 0
        bufIdx = 0
        for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            #pid_m = pid % num_pid_m
            #pid_n = pid // num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_k = 0
            accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                b0_buf = txl.get_buffer(b0, bufIdx)
                txl.mbar_wait(mbar_p_b0, phase)
                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)
                    txl.mbar_wait(mbar_p_a0, phase)
                    accumulator = tl.dot(a0_buf, b0_buf.T, accumulator) # accumulator is reg, no contention among buffers
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c1)
                if txl.is_warpgroup([2]): # TODO: else test
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)
                    txl.mbar_wait(mbar_p_a1, phase)
                    accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    txl.dot_wait(0)
                    txl.mbar_arrive(mbar_c2)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0: # TODO: pipelinestate
                    phase = phase^1

            offs_n = offs_bn + tl.arange(0, BLOCK_SIZE_N)[None, :]

            if bias_ptr is not None:
                bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
                accumulator += bias.to(tl.float32)
            if apply_gelu:
                a:tl.constexpr = 0.797885
                b:tl.constexpr = 0.044715
                x = accumulator
                x_cube = x * x * x
                tanh_res = 2.0 * tl.sigmoid(2.0 * a * (x + b * x_cube)) - 1.0
                accumulator = 0.5 * x * (1.0 + tanh_res)
            if residual_ptr is not None:
                if txl.is_warpgroup([1]):
                    offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M//2)[:, None]
                else:
                    offs_m = offs_am + BLOCK_SIZE_M//2 + tl.arange(0, BLOCK_SIZE_M//2)[:, None]
                mask = (offs_m < M) & (offs_n < N)
                residual = tl.load(residual_ptr + offs_m * N + offs_n, mask=mask, other=0.0)
                accumulator += residual.to(tl.float32)

            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M//2, offs_bn], c)

@torch.no_grad()
def fused_ffn_txl(
    a,
    b,
    bias=None,
    residual=None,
    add_gelu=False,
    dropout_prob=0.0,
):  
    out_shape_0 = a.shape[:-1]
    a = a.view(-1, a.shape[-1])
    b = b.T.contiguous()

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    if bias is not None: # bias: (N,)
        assert bias.is_contiguous()
        assert b.shape[0] == bias.shape[0]
    if residual is not None:
        residual = residual.view(c.shape)
        assert residual.is_contiguous()
    
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    if dtype == torch.float16:
        matmul_persistent_ws_tma_txl_kernel[grid](
            a_desc, b_desc, c_desc,  #
            M, N, K,  #
            FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
            NUM_SMS=NUM_SMS,  #
            WARP_SPECIALIZE=False,  #
            apply_gelu=add_gelu,
            bias_ptr=bias,
            residual_ptr=residual,
        )
    return c.view(*out_shape_0, N)