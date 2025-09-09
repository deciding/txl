from modal import Image, App, Volume
import pathlib
local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"
txl_wheel_file = local_dir / "txl-3.4.0-cp312-cp312-linux_x86_64.whl"

test_file = root_dir / "python" / "txl" / "tutorials" / "02-flash-attention.py"

app = App(name="txl-ce")  # Note: this is optional since Modal 0.57
volume = Volume.from_name("txl-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

txl_image = (
    Image.debian_slim(python_version="3.12")
    #Image.from_dockerfile(path="./Dockerfile")
    .workdir("/workspace")
    .add_local_file(txl_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
    .pip_install_from_requirements(requirements_file) # local file not remote file
    .pip_install("quack-kernels")
    .run_commands(
        "pip install /workspace/txl-3.4.0-cp312-cp312-linux_x86_64.whl",
    )
    .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False) # copy after image build, no need rebuild
)

# Example function that uses the image
@app.function(gpu="H100", image=txl_image, timeout=60,
		volumes={"/workspace/dump": volume})
def test_flash_attention():

    def get_gpu_type():
        import subprocess
    
        try:
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if 'H100' in line and 'HBM3' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return False

    if not get_gpu_type():
        return

    import time

    import torch
    import torch.nn.functional as F
    from triton.testing import do_bench

    import cutlass
    import cutlass.torch as cutlass_torch

    from quack.cross_entropy import _cross_entropy, cross_entropy

    M = 8192
    N = 16384
    dtype = cutlass.BFloat16
    warmup_iterations = 10
    iterations = 100

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)

    loss = _cross_entropy(x, target)

    compiled_func_ref = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'))

    fn = lambda: _cross_entropy(x, target)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)

    # Memory bandwidth calculation: read x (M*N elements) + read target (M elements) + write loss (M elements)
    mem_bytes = (M * N + M + M) * dtype.width // 8
    mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: compiled_func_ref(x, target)
    for _ in range(5): fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Ref kernel execution time: {avg_time:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")


