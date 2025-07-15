import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from contextlib import contextmanager

import txl
import txl.proton

from typing import Optional

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

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

@txl.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir')
#@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_tma_txl_kernel(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
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
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # TODO:
    # 1. a = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, stages=3) # stages to be auto
    # 1. pass accelerate_matmul
    # 1. remove the auto smem alloc
    # 2. mbarAP mbarAC mbarBP mbarBC txl.mbar_alloc(stages, threads) alloc+init
    # 2. txl.mbar_wait(mbar, phase)
    # 2. txl.mbar_arrive(mbar)
    # 2. txl.mbar_expetc_bytes(mbar, phase)
    # 2. TODO pipelinestate, mbar alloc, advance, hide all mbar
    # 3. a[bufIdx] = txl.tma_load(desc, [offs_am, offs_k], mbar)
    # 3. txl.tma_store(desc, [offs_am, offs_bn], smem)
    # 4. txl.dot(a[bufIdx], b[bufIdx], acc)
    # 4. txl.dot_wait(pendings=1) TODO: combine this two


    #if txl.warpgroup_id() == 0:
    #    txl.reg_dealloc(40) #TODO: tobe auto
    #a = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, stages=1)
    #b = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, stages=1)

    a = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)
    b = txl.smem_alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=dtype, num_stages=NUM_STAGES)

    mbar_producer = txl.mbar_alloc(1, num_stages=NUM_STAGES)
    #b = txl.local_alloc([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=dtype)
    phase = 0
    bufIdx = 0
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            cur_mbar = txl.get_buffer(mbar_producer, bufIdx)

            txl.tma_load(
                a,
                a_desc_ptr,
                [offs_am, offs_k],
                cur_mbar,
            )
            txl.tma_load(
                b,
                b_desc_ptr,
                [offs_bn, offs_k],
                cur_mbar,
            )

            txl.mbar_expect(cur_mbar, BLOCK_SIZE_M*BLOCK_SIZE_K*2 + BLOCK_SIZE_N*BLOCK_SIZE_K*2)
            txl.mbar_wait(cur_mbar, phase)

            #accumulator = tl.dot(a[0], b[0].T, accumulator)
            #a = tl._experimental_descriptor_load(
            #    a_desc_ptr,
            #    [offs_am, offs_k],
            #    [BLOCK_SIZE_M, BLOCK_SIZE_K],
            #    dtype,
            #)
            #b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)
            accumulator = tl.dot(a, b.T, accumulator)

            offs_k += BLOCK_SIZE_K
            bufIdx = (bufIdx + 1) % NUM_STAGES
            phase = phase if bufIdx > 0 else phase^1

        c = accumulator.to(dtype)
        tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])

    #if txl.warpgroup_id == 1:
    #    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
    #        group_id = pid // num_pid_in_group
    #        first_pid_m = group_id * GROUP_SIZE_M
    #        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    #        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    #        pid_n = (pid % num_pid_in_group) // group_size_m

    #        offs_am = pid_m * BLOCK_SIZE_M
    #        offs_bn = pid_n * BLOCK_SIZE_N
    #        offs_k = 0

    #        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    #        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    #            accumulator = tl.dot(a, b.T, accumulator)
    #            offs_k += BLOCK_SIZE_K

    #    c = accumulator.to(dtype)
    #    tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])


def matmul_persistent_tma_txl(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            a.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_K"],
            a.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            b.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            b.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_N"],
            c.element_size(),
        )
        res = (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )
        return res

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    matmul_persistent_tma_txl_kernel[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
    )
    return c


#@txl.autotune(
#    configs=[
#        triton.Config(
#            {
#                "BLOCK_SIZE_M": 128,
#                "BLOCK_SIZE_N": 256,
#                "BLOCK_SIZE_K": 64,
#                "GROUP_SIZE_M": 8,
#                "NUM_CONSUMER_GROUPS": 2,
#            },
#            num_stages=2,
#            num_warps=4,
#            num_consumer_groups=2,
#            num_buffers_warp_spec=3,
#        ),
#    ],
#    key=["M", "N", "K"],
#    use_cuda_graph=True,
#)
@txl.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@txl.jit(launch_metadata=_matmul_launch_metadata, diff_mode='ttgir', use_txl=False)
#@txl.jit(launch_metadata=_matmul_launch_metadata)
def matmul_persistent_tma_ws_cooperative_kernel(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl._experimental_descriptor_load(
                a_desc_ptr,
                [offs_am, offs_k],
                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                dtype,
            )
            b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)

            accumulator = tl.dot(a, b.T, accumulator)
            offs_k += BLOCK_SIZE_K

        c = accumulator.to(dtype)
        tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])


def matmul_persistent_tma_ws_cooperative(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            a.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_K"],
            a.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            b.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            b.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_N"],
            c.element_size(),
        )
        res = (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )
        return res

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    matmul_persistent_tma_ws_cooperative_kernel[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
    )
    return c


def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(reps, warmup_reps, fn, *args):
    for _ in range(warmup_reps):
        fn(*args)
    with txl.proton.record():
        for _ in range(reps):
            fn(*args)


def bench(K, dtype, reps=1000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    if cublas is not None:
        bench_fn(reps, warmup_reps, cublas_matmul, a, b)
    #if dtype == torch.float16:
    #    bench_fn(reps, warmup_reps, torch_matmul, a, b)
    #bench_fn(reps, warmup_reps, matmul, a, b.T)
    #bench_fn(reps, warmup_reps, matmul_persistent, a, b.T)
    if supports_tma():
        #bench_fn(reps, warmup_reps, matmul_tma_persistent, a, b)
        #bench_fn(reps, warmup_reps, matmul_descriptor_persistent, a, b)

        #bench_fn(reps, warmup_reps, matmul_persistent_tma_ws_cooperative, a, b)
        bench_fn(reps, warmup_reps, matmul_persistent_tma_txl, a, b)


def validate0(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    cublas_result = cublas_matmul(a, b) if cublas is not None else None

    #matmul_persistent_tma_ws_cooperative_result = matmul_persistent_tma_ws_cooperative(a, b) if supports_tma() else None
    matmul_persistent_tma_ws_cooperative_result = matmul_persistent_tma_txl(a, b) if supports_tma() else None

    print(matmul_persistent_tma_ws_cooperative_result)
    print(cublas_result)
    naive_vs_matmul_persistent_tma_ws_cooperative = "✅" if torch.allclose(
        cublas_result.to(torch.float16), matmul_persistent_tma_ws_cooperative_result.to(torch.float16),
        atol=1.0) else "❌"
    print(f"M={M}, N={N}, K={K} verification naive vs: ", end="")
    print(f"TMA persistent with warp specialization: {naive_vs_matmul_persistent_tma_ws_cooperative} ", end="")

def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    if precision == 'fp8':
        metric_names = ["tflop8/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names
    elif precision == 'fp32':
        metric_names = ["tflop32/s"] + metric_names
    else:
        raise ValueError("the precision must be one of the following: fp8, fp16, fp32")
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
    else:
        dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(0)

        validate0(8192, 8192, args.K_range[0], dtype)

        if True:
            with txl.proton.profile("matmul", precision='fp16') as prof:
                print(f"profile id: {prof.prof_id}")
                for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                    bench(K, dtype, reps=100, warmup_reps=10)
