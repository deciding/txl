"""
Modal script to run FA4 (CuTeDSL) sm100 (Blackwell) benchmark on B200 GPU.
Compares local (mounted) vs pip (official) versions.
"""

from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent.parent
GPU_model = "B200"

app = App(name="fa4-sm100-benchmark")

VOLUME_NAME = "fa4-dump"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

fa4_image = (
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

fa4_image = (
    fa4_image.pip_install("torch", "pytest", "einops")
    .pip_install("nvidia-cutlass-dsl>=4.4.1")
    .pip_install("quack-kernels>=0.2.10")
    .pip_install("apache-tvm-ffi>=0.1.5,<0.2")
    .pip_install("torch-c-dlpack-ext")
    .pip_install("triton==3.5.1")
    .pip_install("flash-attn-4==4.0.0b4")
    .add_local_dir(
        root_dir / "docker" / "flash_attn", remote_path="/workspace/flash_attention"
    )
)


@app.function(
    gpu=GPU_model,
    image=fa4_image,
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_fa4_benchmark():
    import torch
    import sys
    from typing import NamedTuple
    from triton.testing import do_bench
    import os

    from datetime import datetime
    dump_name = "fa4" + "".join(str(datetime.now()).replace(':', '.').split())
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = DUMP_DIR
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_LINEINFO"] = "1"



    class Timing(NamedTuple):
        mean: float

    def time_fwd(func, *args, repeats=30, **kwargs):
        return Timing(
            do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3
        )

    def calc_tflops(
        time_ms, batch_size, nheads, seqlen_q, seqlen_k, head_dim, causal=False
    ):
        avg_seqlen = seqlen_k if not causal else (seqlen_k - seqlen_q + seqlen_k) // 2
        flops = batch_size * nheads * 2 * seqlen_q * avg_seqlen * (head_dim + head_dim)
        return flops / time_ms / 1e12

    print("=" * 60)
    print("FA4 sm100 (Blackwell) Benchmark - Local vs Pip Comparison")
    print("=" * 60)

    # Check GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    # Config: seq_len=8k, head_dim=128, batch=4 (matching official benchmark)
    batch_size = 4
    nheads = 16
    seqlen_q = 8192
    seqlen_k = 8192
    head_dim = 128
    dtype = torch.bfloat16
    causal = False
    repeats = 30

    batch_size = 1
    nheads = 16
    seqlen_q = 128
    seqlen_k = 128
    head_dim = 128
    repeats = 1

    print(
        f"\nConfig: batch={batch_size}, heads={nheads}, seq_len={seqlen_q}, head_dim={head_dim}, causal={causal}"
    )
    print(f"Dtype: {dtype}, Repeats: {repeats}")

    # Create inputs
    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, seqlen_k, nheads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, seqlen_k, nheads, head_dim, dtype=dtype, device="cuda")

    # ===== Import Local (Mounted) Version =====
    print("\n" + "=" * 60)
    print("=== Local (Mounted) FA4 ===")
    print("=" * 60)

    sys.path.insert(0, "/workspace/flash_attn")
    from flash_attn.cute import interface as interface_local

    flash_attn_func_local = interface_local.flash_attn_func

    # Warmup
    for _ in range(5):
        _ = flash_attn_func_local(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # Benchmark
    m_local = time_fwd(flash_attn_func_local, q, k, v, causal=causal, repeats=repeats)
    tflops_local = calc_tflops(
        m_local.mean, batch_size, nheads, seqlen_q, seqlen_k, head_dim, causal
    )

    print(f"Mean time: {m_local.mean * 1e3:.3f} ms")
    print(f"TFLOPS: {tflops_local:.2f}")

    # ===== Import Pip (Official) Version =====
    print("\n" + "=" * 60)
    print("=== Pip (Official) FA4 ===")
    print("=" * 60)

    # Remove local path temporarily to import pip version
    sys.path.remove("/workspace/flash_attn")

    # Force reimport
    if "flash_attn" in sys.modules:
        del sys.modules["flash_attn"]
    if "flash_attn.cute" in sys.modules:
        del sys.modules["flash_attn.cute"]
    if "flash_attn.cute.interface" in sys.modules:
        del sys.modules["flash_attn.cute.interface"]

    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_pip

    # Restore local path
    sys.path.insert(0, "/workspace/flash_attn")

    # Warmup
    for _ in range(5):
        _ = flash_attn_func_pip(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # Benchmark
    m_pip = time_fwd(flash_attn_func_pip, q, k, v, causal=causal, repeats=repeats)
    tflops_pip = calc_tflops(
        m_pip.mean, batch_size, nheads, seqlen_q, seqlen_k, head_dim, causal
    )

    print(f"Mean time: {m_pip.mean * 1e3:.3f} ms")
    print(f"TFLOPS: {tflops_pip:.2f}")

    # ===== Comparison =====
    print("\n" + "=" * 60)
    print("=== Comparison ===")
    print("=" * 60)
    print(f"Local (Mounted): {tflops_local:.2f} TFLOPS")
    print(f"Pip (Official):  {tflops_pip:.2f} TFLOPS")
    diff = tflops_local - tflops_pip
    diff_pct = (diff / tflops_pip) * 100 if tflops_pip != 0 else 0
    print(f"Difference:      {diff:+.2f} TFLOPS ({diff_pct:+.2f}%)")

    # Save results
    results_file = "/workspace/dump/fa4_benchmark_results.txt"
    with open(results_file, "w") as f:
        f.write(f"FA4 sm100 Benchmark - Local vs Pip Comparison\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(
            f"Config: batch={batch_size}, heads={nheads}, seq_len={seqlen_q}, head_dim={head_dim}, causal={causal}\n"
        )
        f.write(f"\n")
        f.write(f"Local (Mounted): {tflops_local:.2f} TFLOPS\n")
        f.write(f"Pip (Official):  {tflops_pip:.2f} TFLOPS\n")
        f.write(f"Difference:      {diff:+.2f} TFLOPS ({diff_pct:+.2f}%)\n")

    print(f"\nResults saved to {results_file}")
    print("Done!")
    print(f"to download and view: modal volume get {VOLUME_NAME} {dump_name}")


@app.local_entrypoint()
def main():
    run_fa4_benchmark.remote()
