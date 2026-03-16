"""
Modal script to benchmark CUTLASS CuTeDSL dense GEMM on Blackwell B200 GPU.
Tests torch.matmul and dense_gemm with same settings.
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
    # .pip_install("jax", "jaxlib")
    .pip_install("triton==3.5.1")
    .add_local_dir(
        root_dir / "docker" / "tutorials" / "cuteDSL",
        remote_path="/workspace/cuteDSL",
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
    import os
    import subprocess
    from datetime import datetime

    dump_name = "dense_gemm" + "".join(str(datetime.now()).replace(":", ".").split())
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)

    DEVICE = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    import cutlass

    blackwell_dst = "/workspace/cuteDSL/blackwell"

    M = 8192
    N = 8192
    K = 4096
    warmup = 10
    repeats = 100

    print(f"\n=== Benchmark: {M}x{N}x{K} ===")
    print(f"Warmup: {warmup}, Iterations: {repeats}")

    import torch.utils.benchmark as benchmark

    # 1. torch.matmul (uses cuBLAS under the hood)
    print("\n=== 1. torch.matmul Benchmark ===")
    torch.manual_seed(1111)
    a = torch.empty(M, K, dtype=torch.float16).random_(-2, 2).to("cuda")
    b = torch.empty(N, K, dtype=torch.float16).random_(-2, 2).to("cuda")

    timer = benchmark.Timer(
        stmt="torch.matmul(a, b.T)",
        globals={"a": a, "b": b},
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    avg_time_ms = result.mean * 1e3
    flops = 2.0 * M * N * K
    tflops = flops / (avg_time_ms * 1e-3) / 1e12

    print(f"torch.matmul: {avg_time_ms:.4f} ms, {tflops:.2f} TFLOPS")
    print(f"  (median: {result.median * 1e3:.4f} ms)")

    # 2. dense_gemm.py via subprocess
    print("\n=== 2. dense_gemm.py Benchmark ===")

    TERMINAL = False
    if TERMINAL:
        result = subprocess.run(
            [
                "python",
                "/workspace/cuteDSL/blackwell/dense_gemm.py",
                "--mnkl",
                f"{M},{N},{K},1",
                "--ab_dtype",
                "Float16",
                "--acc_dtype",
                "Float32",
                "--c_dtype",
                "Float16",
                "--mma_tiler_mn",
                "256,256",
                "--cluster_shape_mn",
                "2,1",
                "--use_2cta_instrs",
                "--use_tma_store",
                "--warmup_iterations",
                str(warmup),
                "--iterations",
                str(repeats),
                "--skip_ref_check",
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    else:
        from cuteDSL.blackwell.dense_gemm import run

        us = run(
            (M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            use_cold_l2=False,
        )
        time_ms = us / 1000
        tflops = flops / time_ms / 1e9
        print(f"dense_gemm: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 3. dense_gemm_1.py (low-level mbarrier API)
    print("\n=== 3. dense_gemm_1.py Benchmark ===")
    from cuteDSL.blackwell.dense_gemm_1 import run_dense_gemm as run_dense_gemm_1

    us = run_dense_gemm_1(
        (M, N, K),
        tolerance=0.1,
        warmup_iterations=warmup,
        iterations=repeats,
        skip_ref_check=False,
    )
    time_ms = us / 1000
    tflops1 = flops / time_ms / 1e9
    print(f"dense_gemm_1: {time_ms:.4f} ms, {tflops1:.2f} TFLOPS")

    # 4. dense_gemm_2.py (Pipeline API)
    print("\n=== 4. dense_gemm_2.py Benchmark ===")
    from cuteDSL.blackwell.dense_gemm_2 import run_dense_gemm as run_dense_gemm_2

    us = run_dense_gemm_2(
        (M, N, K),
        tolerance=0.1,
        warmup_iterations=warmup,
        iterations=repeats,
        skip_ref_check=False,
    )
    time_ms = us / 1000
    tflops2 = flops / time_ms / 1e9
    print(f"dense_gemm_2: {time_ms:.4f} ms, {tflops2:.2f} TFLOPS")

    print(f"\nDone! Results saved to: {DUMP_DIR}")


@app.local_entrypoint()
def main():
    run_dense_gemm.remote()
