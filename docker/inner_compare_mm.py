import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from contextlib import contextmanager

from typing import Optional
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
        def jit(use_txl=False, diff_mode='ttir', diff_select=-1, log_dir='', src_file='', launch_metadata=None):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def autotune(configs=[], key='', use_cuda_graph=False):
            def decorator(func):
                return func
            return decorator
    Has_TXL = False
    DEVICE = torch.device('cuda:0')
    print("No txl")

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



@txl.autotune(
    configs=[
        txl.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                #"BLOCK_SIZE_N": 128,
                #"BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
            num_warpgroups=3,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
#@txl.jit(launch_metadata=_matmul_launch_metadata)
@txl.jit(launch_metadata=_matmul_launch_metadata, src_file=f'./matmul_persistent_ws_tma_txl_kernel.ptx')
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
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
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

    tid = txl.tid(0)
    c_add = tid * 0.0001
    c_add = c_add.to(tl.float16)


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
                txl.mbar_expect(mbar_p_a0, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a0_buf, a_desc, [offs_am, offs_k], mbar_p_a0)

                txl.mbar_wait(mbar_c2, phase)
                txl.mbar_expect(mbar_p_a1, BLOCK_SIZE_M//2*BLOCK_SIZE_K*2)
                txl.tma_load(a1_buf, a_desc, [offs_am + BLOCK_SIZE_M // 2, offs_k], mbar_p_a1)


                txl.mbar_expect(mbar_p_b0, BLOCK_SIZE_N*BLOCK_SIZE_K*2)
                txl.tma_load(b0_buf, b_desc, [offs_bn, offs_k], mbar_p_b0)


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
            #accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float32)
            accumulator = tl.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=tl.float16)

            if txl.is_warpgroup([1]):
                txl.bar_arrive(8, 256)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                mbar_p_b0 = txl.get_buffer(mbar_producer_b0, bufIdx)

                b0_buf = txl.get_buffer(b0, bufIdx)

                txl.mbar_wait(mbar_p_b0, phase)


                if txl.is_warpgroup([1]):
                    mbar_p_a0 = txl.get_buffer(mbar_producer_a0, bufIdx)
                    mbar_c1 = txl.get_buffer(mbar_consumer1, bufIdx)
                    a0_buf = txl.get_buffer(a0, bufIdx)

                    txl.bar_wait(8, 256)

                    pl.enter_scope("wait1")
                    txl.mbar_wait(mbar_p_a0, phase)
                    pl.exit_scope("wait1")


                    #pl.enter_scope("dot1")

                    #accumulator = tl.dot(a0_buf, b0_buf.T, accumulator) # accumulator is reg, no contention among buffers
                    #txl.dot_wait(0)

                    with pl.scope("x1"):
                        accumulator = a0_buf + c_add
                        accumulator = accumulator + c_add
                        accumulator = accumulator + c_add
                        accumulator = accumulator + c_add
                    #with pl.scope("fma"):
                    #    a = a0_buf * 2.0 + 0.0001
                    #with pl.scope("add"):
                    #    a += accumulator
                    #with pl.scope("cast"):
                    #    a = a.to(tl.float32)
                    #with pl.scope("ex2_1"):
                    #    accumulator = tl.exp2(a)
                    #with pl.scope("ex2_2"):
                    #    accumulator = tl.exp2(accumulator)
                    #with pl.scope("ex2_3"):
                    #    accumulator = tl.exp2(accumulator)

                    #pl.exit_scope("dot1")

                    txl.bar_arrive(9, 256)

                    txl.mbar_arrive(mbar_c1)

                if txl.is_warpgroup([2]): # TODO: else test
                    mbar_p_a1 = txl.get_buffer(mbar_producer_a1, bufIdx)
                    mbar_c2 = txl.get_buffer(mbar_consumer2, bufIdx)
                    a1_buf = txl.get_buffer(a1, bufIdx)

                    txl.bar_wait(9, 256)

                    pl.enter_scope("wait2")
                    txl.mbar_wait(mbar_p_a1, phase)
                    pl.exit_scope("wait2")


                    #pl.enter_scope("dot2")

                    #accumulator = tl.dot(a1_buf, b0_buf.T, accumulator)
                    #txl.dot_wait(0)

                    with pl.scope("x2"):
                        accumulator = a1_buf + c_add
                        accumulator = accumulator + c_add
                        accumulator = accumulator + c_add
                        accumulator = accumulator + c_add

                    #a = a1_buf * 2.0 + 0.0001 + accumulator # 128x128 x 2fma + add
                    #a = a.to(tl.float32) # 128x128 x 0.5 cast
                    #accumulator = tl.exp2(a) # 128x128 x4
                    #accumulator = tl.exp2(accumulator+0.0001) #128x128 5
                    #accumulator = tl.exp2(accumulator+0.0001) # 128x128 5

                    #pl.exit_scope("dot2")

                    txl.bar_arrive(8, 256)

                    txl.mbar_arrive(mbar_c2)

                offs_k += BLOCK_SIZE_K
                bufIdx = (bufIdx + 1) % NUM_STAGES
                if bufIdx == 0: # TODO: pipelinestate
                    phase = phase^1

            c = accumulator.to(dtype)
            if txl.is_warpgroup([1]):
                c_desc.store([offs_am, offs_bn], c)
            if txl.is_warpgroup([2]):
                c_desc.store([offs_am + BLOCK_SIZE_M//2, offs_bn], c)

def matmul_tma_ws_persistent_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

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

    profiling = True
    if profiling:
        proton.start("dump/mm", backend="instrumentation", mode='default:sampling_strategy=selective:sampling_options=0,4,8', data="trace")
    matmul_persistent_ws_tma_txl_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=False,  #
    )

    if profiling:
        proton.finalize()

    return c

def validate(M, N, K, dtype, log=False, algo='0'):
    print(f"{M=}, {N=}, {K=}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    bn = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = bn.T.contiguous()
    matmul_tma_ws_persistent_txl(a, b)

from triton import knobs
#os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
#os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
#os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-wgmma-pipeline"
#knobs.runtime.override_arch='sm100'
knobs.autotuning.print=True
knobs.compilation.always_compile=True
#dump_dir = '/workspace/dump'
dump_dir = None

if dump_dir:
    knobs.compilation.dump_ir=True
    knobs.cache.dump_dir=dump_dir
validate(1024, 1024, 1024, torch.float16)
