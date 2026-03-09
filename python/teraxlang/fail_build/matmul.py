from modal import Image, App, Volume
import pathlib
import os

local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"

Use_TXL = True
GPU_model = "H100"  # B200
# GPU_model = 'B200' # B200
app_name = "txl" if Use_TXL else "triton"

txl_wheel_name = os.environ.get("TXL_WHEEL_NAME")

use_pip = False
if not txl_wheel_name:
    dist_dir = root_dir / "thirdparty" / "triton" / "dist"
    # Search for both teraxlang-*.whl and txl-*.whl
    wheel_files = list(dist_dir.glob("teraxlang-*.whl")) + list(
        dist_dir.glob("txl-*.whl")
    )
    if wheel_files:
        txl_wheel = next(
            (f for f in wheel_files if "manylinux" in f.name), wheel_files[0]
        )
        txl_wheel_name = txl_wheel.name
        print(f"Using wheel: {txl_wheel_name}")
        main_cmd = f"pip install /workspace/{txl_wheel_name}"
    else:
        # txl_wheel_name = "teraxlang-3.5.1-cp312-cp312-linux_x86_64.whl"
        use_pip = True
        print("Using Pip")
        main_cmd = f"pip install teraxlang"


if txl_wheel_name:
    txl_wheel_file = root_dir / "thirdparty" / "triton" / "dist" / txl_wheel_name
else:
    txl_wheel_file = None

dump_dir = "modal_dump"

test_file = root_dir / "python" / "teraxlang" / "tutorials" / "01-matmul.py"
# ptx_file_name = "matmul_persistent_tma_txl_bw_kernel6"
# ptx_file = local_dir / dump_dir / f"{ptx_file_name}.ptx"
# signature_file = local_dir / dump_dir / f"{ptx_file_name}_signature.json"
# json_file = local_dir / dump_dir / f"{ptx_file_name}.json"

# txl
app = App(name=f"{app_name}-matmul")  # Note: this is optional since Modal 0.57
volume = Volume.from_name(
    f"{app_name}-matmul-dump", create_if_missing=True
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
        .workdir("/workspace")
    )
    # txl
    if txl_wheel_file:
        txl_image = txl_image.add_local_file(
            txl_wheel_file, remote_path="/workspace/", copy=True
        )  # copy the local code to the image
    txl_image = (
        txl_image.run_commands("ls .")
        .pip_install_from_requirements(requirements_file)  # local file not remote file
        # txl
        .run_commands(
            # f"pip install /workspace/{txl_wheel_name}",
            main_cmd
        )
        .env(
            {
                "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib"
            }
        )
        .add_local_file(
            test_file, remote_path="/workspace/test_txl.py", copy=False
        )  # copy after image build, no need rebuild
        # .add_local_file(
        #    ptx_file, remote_path=f"/workspace/{ptx_file_name}.ptx", copy=False
        # )
        # .add_local_file(
        #    signature_file,
        #    remote_path=f"/workspace/{ptx_file_name}_signature.json",
        #    copy=False,
        # )
        # .add_local_file(
        #    json_file, remote_path=f"/workspace/{ptx_file_name}.json", copy=False
        # )
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
    gpu=GPU_model, image=txl_image, timeout=300, volumes={"/workspace/dump": volume}
)
def test_flash_attention(
    dump_dir: str = "default",
    use_ncu: bool = False,
    ncu_metrics: str = "sm__throughput.avg.pct_of_peak_sustained",
):

    def find_440():
        import subprocess

        try:
            result = subprocess.run(
                ["find", "/", "-name", "nvidia*.ko"],
                capture_output=True,
                text=True,
                check=True,
            )
            print("4.4.0 locations found:")
            print(result.stdout)

            paths = [p for p in result.stdout.strip().split("\n") if p]
            return paths[0] if paths else None
        except subprocess.CalledProcessError:
            return None

    def find_modprobe():
        import subprocess

        try:
            result = subprocess.run(
                ["find", "/", "-name", "modprobe"],
                capture_output=True,
                text=True,
                check=True,
            )
            print("modprobe locations found:")
            print(result.stdout)

            paths = [p for p in result.stdout.strip().split("\n") if p]
            return paths[0] if paths else None
        except subprocess.CalledProcessError:
            return None

    def find_ncu():
        import subprocess

        try:
            result = subprocess.run(
                ["find", "/", "-name", "ncu"],
                capture_output=True,
                text=True,
                check=True,
            )
            print("NCU locations found:")
            print(result.stdout)


            # Prefer /usr/local/cuda-12.4/bin/ncu or the one without "target" in path
            ncu_paths = [p for p in result.stdout.strip().split("\n") if p]
            for path in ncu_paths:
                if "cuda-12.4" in path:
                    return path
            for path in ncu_paths:
                if "target" not in path:
                    return path
            return ncu_paths[0] if ncu_paths else None
        except subprocess.CalledProcessError:
            return None

    def get_gpu_type():
        import subprocess

        try:
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(
                ["nvidia-smi", "-q"], capture_output=True, text=True, check=True
            )
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    # if 'H100' in line and 'HBM3' in line:
                    if GPU_model in line:
                        if GPU_model == "H100" and "HBM3" not in line:
                            return False
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print(
                "nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH."
            )
        return False

    def import_cuBLAS_lib():
        import os, ctypes, pathlib
        import nvidia.cublas, nvidia.cuda_runtime

        cublas_dir = (pathlib.Path(nvidia.cublas.__file__).parent / "lib").resolve()
        cudart_dir = (
            pathlib.Path(nvidia.cuda_runtime.__file__).parent / "lib"
        ).resolve()

        os.environ["LD_LIBRARY_PATH"] = f"{cublas_dir}:{cudart_dir}:" + os.environ.get(
            "LD_LIBRARY_PATH", ""
        )

        for name12, name in [
            ("libcublas.so.12", "libcublas.so"),
            ("libcublasLt.so.12", "libcublasLt.so"),
        ]:
            src = cublas_dir / name12
            dst = cublas_dir / name
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(name12)
                except FileExistsError:
                    pass
                except PermissionError:
                    pass

        ctypes.CDLL(str(cudart_dir / "libcudart.so.12"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(cublas_dir / "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(cublas_dir / "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)

    if not get_gpu_type():
        return
    import_cuBLAS_lib()

    from test_txl import test_matmul
    import subprocess

    ncu_path = find_ncu() if use_ncu else None

    if use_ncu and ncu_path:
        print(f"Running with NCU: {ncu_path}")
        print(f"Metrics: {ncu_metrics}")

        # Ensure dump directory exists
        dump_path = f"/workspace/dump/{dump_dir}"
        os.makedirs(dump_path, exist_ok=True)

        # Warmup: run matmul once to compile the kernel (required for NCU to profile)
        print("Warmup: Running matmul to compile kernels...")
        test_matmul(dump_path, "hopper_txl_ws_persistent")
        print("Warmup complete, now profiling with NCU...")

        four_path = find_440() if use_ncu else None
        print(f"Running nvidia-smi command: ")
        result = subprocess.run([
            "nvidia-smi"
            ], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

        modprobe_path = find_modprobe() if use_ncu else None
        print(f"Running modprobe command: ")
        result = subprocess.run([
            modprobe_path,
            "nvidia"
            ], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        return

        cmd = [
            ncu_path,
            "--metrics",
            #ncu_metrics,
            "sm__cycles_active.avg",
            "python",
            "/workspace/test_txl.py"
            #"-c",
            #"import sys; sys.path.insert(0, '/workspace'); from test_txl import test_matmul; test_matmul('"
            #+ dump_path
            #+ "', 'hopper_txl_ws_persistent')",
        ]
        print(f"Running NCU command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    else:
        if use_ncu:
            print("NCU requested but not found in container")
        # test_matmul("/workspace/dump", "hopper_triton_ws_persistent")
        test_matmul(f"/workspace/dump/{dump_dir}", "hopper_txl_ws_persistent")


@app.local_entrypoint()
def main(
    dump_dir: str = "default",
    use_ncu: bool = False,
    ncu_metrics: str = "sm__throughput.avg.pct_of_peak_sustained",
):
    print(f"Running matmul test with dump_dir: {dump_dir}, use_ncu: {use_ncu}")
    test_flash_attention.remote(dump_dir, use_ncu, ncu_metrics)
