"""
Modal script to run CUTLASS CuTeDSL dense GEMM on Blackwell B200 GPU.
Tests correctness and benchmarks performance.
"""

from datetime import datetime
from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent.parent
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
    .pip_install("triton==3.5.1")
    .add_local_dir(
        root_dir / "thirdparty" / "cutlass" / "examples" / "python" / "CuTeDSL",
        remote_path="/workspace/cutlass",
    )
)


@app.function(
    gpu=GPU_model,
    image=cutlass_image,
    timeout=600,
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

    # Add CuTeDSL to path
    sys.path.insert(0, "/workspace/cutlass")

    M = 8192
    N = 8192
    K = 8192

    print(f"\n=== Dense GEMM Test ===")
    print(f"M={M}, N={N}, K={K}")

    # Allocate tensors
    a = torch.rand(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.rand(K, N, device=DEVICE, dtype=torch.float16)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float16)

    # Reference result using torch
    ref_c = torch.matmul(a, b)

    print("\n=== Running CuTeDSL Dense GEMM ===")

    # Import and run the dense gemm example
    from cutlass.blackwell.dense_gemm import DenseGemmKernel
    import cutlass

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
    gemm(a, b, c, torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Check result
    max_diff = (c - ref_c).abs().max().item()
    print(f"Max diff: {max_diff}")

    if max_diff < 1e-3:
        print("✓ Correctness PASSED!")
    else:
        print("✗ Correctness FAILED!")
        return

    # Benchmark performance
    print("\n=== Performance Benchmark ===")
    warmup = 10
    repeats = 100

    for _ in range(warmup):
        gemm(a, b, c, torch.cuda.current_stream())
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeats):
        gemm(a, b, c, torch.cuda.current_stream())
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time_to(end_event)
    avg_time = elapsed_ms / repeats

    # Compute performance
    flops = 2.0 * M * N * K
    tflops = flops / (avg_time * 1e-3) / 1e12

    print(f"Average time: {avg_time:.2f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")

    print(f"\nDone! Results saved to: {DUMP_DIR}")
    print(f"to download: modal volume get {VOLUME_NAME} {dump_name}")


@app.local_entrypoint()
def main():
    run_dense_gemm.remote()
