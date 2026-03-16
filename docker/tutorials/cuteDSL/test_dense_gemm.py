"""
Modal script to run native CUTLASS CuTeDSL dense_gemm.py on Blackwell B200 GPU.
Uses the native test/benchmark in dense_gemm.py directly.
"""

from datetime import datetime
from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent.parent.parent.parent
GPU_model = "B200"

app = App(name="cutlass-dense-gemm-native")

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
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_dense_gemm_native(
    m: int = 8192,
    n: int = 8192,
    k: int = 4096,
    mma_tiler_m: int = 256,
    mma_tiler_n: int = 256,
    cluster_m: int = 2,
    cluster_n: int = 1,
    use_2cta: bool = True,
    use_tma_store: bool = True,
    warmup: int = 10,
    iterations: int = 100,
    skip_ref_check: bool = False,
):
    """
    Run native dense_gemm.py test with proper benchmarking.
    """
    import torch
    import sys
    import os
    from datetime import datetime

    dump_name = "dense_gemm_native" + "".join(
        str(datetime.now()).replace(":", ".").split()
    )
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)

    DEVICE = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    import cutlass
    import cutlass.pipeline as pipeline
    import cutlass.cute as cute

    cutlass_path = os.path.dirname(cutlass.__file__)
    print(f"=== Debug: cutlass installed at {cutlass_path} ===")

    blackwell_src = "/workspace/cuteDSL/blackwell"
    blackwell_dst = os.path.join(cutlass_path, "blackwell")

    if not os.path.exists(blackwell_dst):
        os.symlink(blackwell_src, blackwell_dst)
        print(f"=== Created symlink: {blackwell_dst} -> {blackwell_src} ===")

    blackwell_init = os.path.join(blackwell_dst, "__init__.py")
    if not os.path.exists(blackwell_init):
        with open(blackwell_init, "w") as f:
            pass

    print(f"=== Debug: contents of {cutlass_path} ===")
    print(os.listdir(cutlass_path))

    print(f"\n=== Dense GEMM Native Test ===")
    print(f"M={m}, N={n}, K={k}")
    print(f"MMA Tiler=({mma_tiler_m}, {mma_tiler_n})")
    print(f"Cluster=({cluster_m}, {cluster_n})")
    print(f"2CTA={use_2cta}, TMA Store={use_tma_store}")
    print(f"Warmup={warmup}, Iterations={iterations}")

    print("\n=== Running native dense_gemm.py ===")
    from cutlass.blackwell.dense_gemm import run

    # Call the native run function - skip reference check for benchmarking
    exec_time = run(
        mnkl=(1, m, n, k),
        ab_dtype=cutlass.Float16,
        c_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        a_major="K",
        b_major="K",
        c_major="R",
        mma_tiler_mn=(mma_tiler_m, mma_tiler_n),
        cluster_shape_mn=(cluster_m, cluster_n),
        use_2cta_instrs=use_2cta,
        use_tma_store=use_tma_store,
        tolerance=1e-1,
        warmup_iterations=warmup,
        iterations=iterations,
        skip_ref_check=True,
    )

    # Calculate TFLOPs
    flops = 2 * m * n * k
    tflops = flops / exec_time / 1e6
    print(f"\n=== Benchmark Results ===")
    print(f"Execution time: {exec_time:.2f} microseconds")
    print(f"TFLOPS: {tflops:.2f}")


@app.local_entrypoint()
def main(
    m: int = 8192,
    n: int = 8192,
    k: int = 4096,
    mma_tiler_m: int = 256,
    mma_tiler_n: int = 256,
    cluster_m: int = 2,
    cluster_n: int = 1,
    use_2cta: bool = True,
    use_tma_store: bool = True,
    warmup: int = 10,
    iterations: int = 100,
):
    run_dense_gemm_native.remote(
        m=m,
        n=n,
        k=k,
        mma_tiler_m=mma_tiler_m,
        mma_tiler_n=mma_tiler_n,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        use_2cta=use_2cta,
        use_tma_store=use_tma_store,
        warmup=warmup,
        iterations=iterations,
    )
