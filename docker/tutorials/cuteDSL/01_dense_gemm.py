"""
Modal script to run CUTLASS CuTeDSL dense GEMM on Blackwell B200 GPU.
Tests correctness and benchmarks performance.
"""

from datetime import datetime
from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent.parent.parent.parent
GPU_model = "B200"

app = App(name="cutlass-dense-gemm")

VOLUME_NAME = "cutlass-dump"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

cutlass_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg", "git")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-12-6")
    .workdir("/workspace")
)

cutlass_image = (
    cutlass_image.pip_install("torch", "pytest")
    .pip_install("nvidia-cutlass-dsl>=4.4.1")
    .pip_install("jax", "jaxlib")
    .pip_install("triton==3.5.1")
    .add_local_dir(
        root_dir / "docker" / "tutorials" / "cuteDSL",
        remote_path="/workspace/cuteDSL",
    )
)


@app.function(
    gpu=GPU_model,
    image=cutlass_image,
    timeout=60,
    volumes={"/workspace/dump": volume},
)
def run_dense_gemm():
    import torch
    import sys
    import os
    from datetime import datetime

    dump_name = "dense_gemm" + "".join(str(datetime.now()).replace(":", ".").split())
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)

    DEVICE = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    # Import cutlass from pip - we'll extend this with blackwell
    import cutlass
    import cutlass.pipeline as pipeline
    import cutlass.cute as cute

    import os
    import sys

    # Find where pip's cutlass is installed
    cutlass_path = os.path.dirname(cutlass.__file__)
    print(f"=== Debug: cutlass installed at {cutlass_path} ===")

    # Create symlink from pip's cutlass to local blackwell
    blackwell_src = "/workspace/cuteDSL/blackwell"
    blackwell_dst = os.path.join(cutlass_path, "blackwell")

    if not os.path.exists(blackwell_dst):
        os.symlink(blackwell_src, blackwell_dst)
        print(f"=== Created symlink: {blackwell_dst} -> {blackwell_src} ===")

    # Also need to add __init__.py to blackwell
    blackwell_init = os.path.join(blackwell_dst, "__init__.py")
    if not os.path.exists(blackwell_init):
        with open(blackwell_init, "w") as f:
            pass  # Empty init file

    # Also need __init__.py for blackwell subpackage
    blackwell_init = os.path.join(blackwell_dst, "__init__.py")
    if not os.path.exists(blackwell_init):
        with open(blackwell_init, "w") as f:
            pass  # Empty init file

    # Now we can import from cutlass.blackwell
    print(f"=== Debug: contents of {cutlass_path} ===")
    print(os.listdir(cutlass_path))

    M = 1024
    N = 1024
    K = 1024

    print(f"\n=== Dense GEMM Test ===")
    print(f"M={M}, N={N}, K={K}")

    print("\n=== Running CuTeDSL Dense GEMM ===")

    # Import and run the dense gemm example
    from cutlass.blackwell.dense_gemm import DenseGemmKernel
    import cutlass
    import cutlass.torch as cutlass_torch

    # Create tensors using cutlass_torch.matrix, then convert to cute tensors
    # This matches the example in dense_gemm.py
    l = 1  # batch size
    a_torch_cpu = cutlass_torch.matrix(
        l, M, K, True, cutlass.Float16
    )  # a_major=True (row major)
    b_torch_cpu = cutlass_torch.matrix(
        l, N, K, False, cutlass.Float16
    )  # b_major=False (col major)
    c_torch_cpu = cutlass_torch.matrix(l, M, N, True, cutlass.Float16)  # c_major=True

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
    )

    # Configure kernel
    acc_dtype = cutlass.Float32
    use_2cta_instrs = True
    mma_tiler_mn = (256, 128)
    cluster_shape_mn = (2, 1)
    use_tma_store = True

    # Create kernel
    gemm = DenseGemmKernel(
        acc_dtype=acc_dtype,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_tma_store=use_tma_store,
    )

    # Run correctness test
    print("\n=== Correctness Test ===")
    # Get CUDA stream - use the same approach as the example
    torch_stream = torch.cuda.current_stream()
    from cuda import bindings

    current_stream = bindings.driver.CUstream(torch_stream.cuda_stream)

    # First compile the kernel
    print("Compiling kernel...")
    compiled_gemm = cute.compile(gemm, a_tensor, b_tensor, c_tensor, current_stream)
    print("Compilation done!")

    # Run the compiled kernel
    compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()

    # Reference result using torch einsum (matching the example exactly)
    # Input tensors are (M, K, L) and (N, K, L)
    ref = torch.einsum(
        "mkl,nkl->mnl",
        a_torch_cpu.to(dtype=torch.float32),
        b_torch_cpu.to(dtype=torch.float32),
    )

    # Convert ref to c_dtype using cute_tensor_like
    _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
        ref, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
    )
    ref_result = ref_torch_gpu.cpu()

    # Copy gpu result back
    kernel_result = c_torch_gpu.cpu()

    # Check result
    print(f"=== Debug: kernel_result shape: {kernel_result.shape} ===")
    print(f"=== Debug: ref_result shape: {ref_result.shape} ===")

    # Compare using torch testing
    torch.testing.assert_close(kernel_result, ref_result, atol=1e-1, rtol=1e-05)
    print("✓ Correctness PASSED!")

    # Benchmark performance
    print("\n=== Performance Benchmark ===")
    warmup = 10
    repeats = 100

    for _ in range(warmup):
        compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeats):
        compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time = elapsed_ms / repeats

    # Compute performance
    flops = 2.0 * M * N * K
    tflops = flops / (avg_time * 1e-3) / 1e12

    print(f"CuTeDSL Average time: {avg_time:.4f} ms")
    print(f"CuTeDSL Performance: {tflops:.2f} TFLOPS")

    # Benchmark PyTorch GEMM for comparison
    print("\n=== PyTorch GEMM Benchmark ===")

    # Prepare tensors for PyTorch (squeeze batch dimension)
    a_torch = a_torch_cpu.squeeze(-1).to("cuda")  # (M, K)
    b_torch = b_torch_cpu.squeeze(-1).to("cuda")  # (N, K) -> transposed below
    c_torch = torch.empty(M, N, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        torch.matmul(a_torch, b_torch.T, out=c_torch)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeats):
        torch.matmul(a_torch, b_torch.T, out=c_torch)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_torch = elapsed_ms / repeats

    tflops_torch = flops / (avg_time_torch * 1e-3) / 1e12

    print(f"PyTorch Average time: {avg_time_torch:.4f} ms")
    print(f"PyTorch Performance: {tflops_torch:.2f} TFLOPS")

    print("\n=== Performance Comparison ===")
    print(f"CuTeDSL: {tflops:.2f} TFLOPS")
    print(f"PyTorch:  {tflops_torch:.2f} TFLOPS")
    print(f"Speedup:  {tflops / tflops_torch:.2f}x")

    print(f"\nDone! Results saved to: {DUMP_DIR}")
    print(f"to download: modal volume get {VOLUME_NAME} {dump_name}")


@app.local_entrypoint()
def main():
    run_dense_gemm.remote()
