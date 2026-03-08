from modal import Image, App, Volume
import pathlib
import os

local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"

Use_TXL = True
app_name = "clock"

txl_wheel_name = os.environ.get("TXL_WHEEL_NAME")
if not txl_wheel_name:
    dist_dir = root_dir / "thirdparty" / "triton" / "dist"
    wheel_files = list(dist_dir.glob("txl-*.whl"))
    if wheel_files:
        txl_wheel = next(
            (f for f in wheel_files if "linux_x86_64" in f.name), wheel_files[0]
        )
        txl_wheel_name = txl_wheel.name
        print(f"Using wheel: {txl_wheel_name}")
    else:
        txl_wheel_name = "txl-3.5.1-cp312-cp312-linux_x86_64.whl"

txl_wheel_file = root_dir / "thirdparty" / "triton" / "dist" / txl_wheel_name

test_file = root_dir / "python" / "txl" / "tutorials" / "01-matmul.py"

ptx_file_name = "matmul_persistent_ws_tma_txl_kernel"
ttgir_file = local_dir / f"{ptx_file_name}.ttgir"  # for proton init
ptx_file = local_dir / f"{ptx_file_name}.ptx"
signature_file = local_dir / f"{ptx_file_name}_signature.json"
json_file = local_dir / f"{ptx_file_name}.json"

# txl
app = App(name=f"{app_name}-compare-matmul")  # Note: this is optional since Modal 0.57
volume = Volume.from_name(
    f"{app_name}-compare-matmul-dump", create_if_missing=True
)  # create a cloud volume to store compiled dump files

if Use_TXL:
    txl_image = (
        Image.debian_slim(python_version="3.12")
        # 2. Install tools required for CUDA repo
        .apt_install("wget", "curl", "gnupg")
        # 3. Add NVIDIA CUDA repository for Debian 12 (Bookworm)
        .run_commands(
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        # 4. Install CUDA debugging tools
        .apt_install(
            "cuda-toolkit-12-4",  # contains cuda-gdb
            # OR if you only want debugger:
            # "cuda-gdb",
        )
        .pip_install(
            [
                "nvidia-ml-py",
                "torch",
            ]
        )
        .workdir("/workspace")
        # txl
        .add_local_file(
            txl_wheel_file, remote_path="/workspace/", copy=True
        )  # copy the local code to the image
        .run_commands("ls .")
        .pip_install_from_requirements(requirements_file)  # local file not remote file
        # txl
        # .run_commands(
        #    "pip install /workspace/txl-3.5.1-cp312-cp312-linux_x86_64.whl",
        # )
        # .env({
        #    "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib"
        # })
        # .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False) # copy after image build, no need rebuild
        # .add_local_file(ttgir_file, remote_path=f"/workspace/{ptx_file_name}.ttgir", copy=False)
        # .add_local_file(ptx_file, remote_path=f"/workspace/{ptx_file_name}.ptx", copy=False)
        # .add_local_file(signature_file, remote_path=f"/workspace/{ptx_file_name}_signature.json", copy=False)
        # .add_local_file(json_file, remote_path=f"/workspace/{ptx_file_name}.json", copy=False)
    )
else:
    txl_image = (
        Image.debian_slim(python_version="3.12")
        # Image.from_dockerfile(path="./Dockerfile")
        .workdir("/workspace")
        .run_commands("ls .")
        .pip_install_from_requirements(requirements_file)  # local file not remote file
        # triton
        .pip_install("triton==3.5.1")
        .env(
            {
                "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib"
            }
        )
        .add_local_file(
            test_file, remote_path="/workspace/test_txl.py", copy=False
        )  # copy after image build, no need rebuild
    )


# Example function that uses the image
@app.function(
    gpu="H100",
    image=txl_image,
    timeout=60,
    # @app.function(gpu="B200", image=txl_image, timeout=300,
    volumes={"/workspace/dump": volume},
)
def test_flash_attention():

    def get_gpu_type():
        import subprocess

        try:
            # result = subprocess.run(['find', '/', '-name', 'libcublas.so*'], capture_output=True, text=True, check=True)
            # output = result.stdout
            # print(output)
            # result = subprocess.run(['find', '/', '-name', 'cuda-gdb'], capture_output=True, text=True, check=True)
            # output = result.stdout
            # print(output)
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(
                ["nvidia-smi", "-q"], capture_output=True, text=True, check=True
            )
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if "H100" in line and "HBM3" in line:
                        # if 'B200' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print(
                "nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH."
            )
        return False

    if not get_gpu_type():
        return

    import torch
    import pynvml
    import time

    def get_h100_clock(handle):
        """Helper to fetch the current graphics clock speed."""
        # 0 = Graphics Clock, 1 = Memory Clock
        return pynvml.nvmlDeviceGetClockInfo(handle, 0)

    def main():
        # 1. Initialize NVML and get GPU handle
        try:
            pynvml.nvmlInit()
            # Index 0 is typically the first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            print(f"Monitoring Device: {gpu_name}")
        except Exception as e:
            print(f"Error initializing NVML: {e}")
            return

        # --- STAGE 1: IDLE ---
        idle_clock = get_h100_clock(handle)
        print(f"Initial Idle Clock: {idle_clock} MHz")

        # --- STAGE 2: WORKLOAD ---
        print("\nStarting H100 Workload (Matrix Multiplication)...")

        # Create large tensors on the H100
        # A 20,000 x 20,000 matrix will definitely wake up the cores
        size = 20000
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")

        # Start math and capture clock speed mid-execution
        # We use a stream to ensure the clock is read while the GPU is busy
        with torch.cuda.device(0):
            # Warm up
            torch.mm(a, b)

            # Actual measurement run
            result = torch.mm(a, b)

            # Read clock while the GPU is processing
            active_clock = get_h100_clock(handle)

            # Ensure completion
            torch.cuda.synchronize()

        print(f"Clock speed DURING workload: {active_clock} MHz")

        # --- STAGE 3: POST-EXECUTION ---
        # Small delay to let the driver settle
        time.sleep(1)
        post_clock = get_h100_clock(handle)
        print(f"Clock speed AFTER execution: {post_clock} MHz")

        # 3. Cleanup
        pynvml.nvmlShutdown()

    main()
