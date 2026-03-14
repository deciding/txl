"""
Modal script to run CUTLASS CuTeDSL dense GEMM on Blackwell B200 GPU.
Tests correctness and benchmarks performance.

Usage:
    # Run specific version (0, 1, 2, or original):
    modal run 01_dense_gemm.py --version 2
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
    timeout=120,
    volumes={"/workspace/dump": volume},
)
def run_dense_gemm(version: str = "2"):
    """
    Run CuTeDSL Dense GEMM on Blackwell B200 GPU.

    Args:
        version: Which version to run:
            - "0": dense_gemm_0.py (4-stage pipelining)
            - "1": dense_gemm_1.py (1-stage pipelining)
            - "2": dense_gemm_2.py (simplified with detailed comments)
            - "original": dense_gemm.py (full-featured)
    """
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
    print(f"Version: {version}")

    if version == "0":
        print("\n=== Running CuTeDSL Dense GEMM (0 - 4-stage pipelining) ===")
        from cutlass.blackwell.dense_gemm_0 import run_dense_gemm as run_gemm

        print("Running GEMM kernel...")
        run_gemm(
            mnk=(M, N, K),
            tolerance=1e-1,
        )
    elif version == "1":
        print("\n=== Running CuTeDSL Dense GEMM (1 - 1-stage pipelining) ===")
        from cutlass.blackwell.dense_gemm_1 import run_dense_gemm as run_gemm

        print("Running GEMM kernel...")
        run_gemm(
            mnk=(M, N, K),
            tolerance=1e-1,
        )
    elif version == "2":
        print(
            "\n=== Running CuTeDSL Dense GEMM (2 - simplified with detailed comments) ==="
        )
        from cutlass.blackwell.dense_gemm_2 import run_dense_gemm as run_gemm

        print("Running GEMM kernel...")
        run_gemm(
            mnk=(M, N, K),
            tolerance=1e-1,
        )
    elif version == "original":
        print("\n=== Running CuTeDSL Dense GEMM (original - full-featured) ===")
        from cutlass.blackwell.dense_gemm import run as run_gemm

        print("Running GEMM kernel...")
        run_gemm(
            mnkl=(1, M, N, K),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="K",
            b_major="K",
            c_major="R",
            tolerance=1e-1,
            skip_ref_check=False,
        )
    else:
        raise ValueError(f"Unknown version: {version}. Choose from: 0, 1, 2, original")

    print("✓ Correctness PASSED!")

    print(f"\nDone! Results saved to: {DUMP_DIR}")


@app.local_entrypoint()
def main(version: str = "2"):
    run_dense_gemm.remote(version=version)
