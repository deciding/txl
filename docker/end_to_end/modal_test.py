from modal import Image, App, Volume
import pathlib
local_dir = pathlib.Path(__file__).parent
requirements_file = local_dir / "requirements.txt"
txl_wheel_file = local_dir / "txl-3.4.0-cp312-cp312-linux_x86_64.whl"

test_file_bench = local_dir / "bench.py"
test_file_gpt = local_dir / "gpt.py"
test_file_kernels = local_dir / "kernels.py"
test_file_test = local_dir / "test.py"
test_file_tmp = local_dir / "tmp.py"
test_file_kernels_attn = local_dir / "kernels_attn.py"
test_file_kernels_ffn = local_dir / "kernels_ffn.py"

app = App(name="txl")  # Note: this is optional since Modal 0.57
volume = Volume.from_name("txl-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

txl_image = (
    Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",  
        add_python="3.12",                       
    )
    #Image.from_dockerfile(path="./Dockerfile")
    .workdir("/workspace")
    .add_local_file(txl_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
    .run_commands( "ls .")
    .pip_install_from_requirements(requirements_file) # local file not remote file
    .pip_install("jaxtyping")
    .run_commands(
        "pip install /workspace/txl-3.4.0-cp312-cp312-linux_x86_64.whl",
    )
    .add_local_file(test_file_bench, remote_path="/workspace/bench.py", copy=False) # copy after image build, no need rebuild
    .add_local_file(test_file_gpt, remote_path="/workspace/gpt.py", copy=False)
    .add_local_file(test_file_kernels, remote_path="/workspace/kernels.py", copy=False)
    .add_local_file(test_file_test, remote_path="/workspace/test_txl.py", copy=False)
    .add_local_file(test_file_tmp, remote_path="/workspace/tmp.py", copy=False)
    .add_local_file(test_file_kernels_attn, remote_path="/workspace/kernels_attn.py", copy=False)
    .add_local_file(test_file_kernels_ffn, remote_path="/workspace/kernels_ffn.py", copy=False)
)

# Example function that uses the image
@app.function(gpu="H100", image=txl_image, timeout=60*60,
		volumes={"/workspace/dump": volume})
def run_demo():
    import subprocess, sys, os, torch, time
    def get_gpu_type():
    
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

    def test_demo(): 
        os.makedirs("/workspace/dump", exist_ok=True)
        logs_dir = pathlib.Path("/workspace/dump/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"mla-{ts}.log"

        env = os.environ.copy()
        # env["TRITON_PRINT_AUTOTUNING"] = "0"
        # env["TRITON_KERNEL_DUMP"] = "1"
        # env["TRITON_DUMP_DIR"] = "/workspace/dump"
        # env["TRITON_ALWAYS_COMPILE"] = "1"
        # env["CUDA_LAUNCH_BLOCKING"] = "1"
        # env["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"

        cmd = [sys.executable, "-u", "/workspace/bench.py"]

        with open(log_path, "w", buffering=1, encoding="utf-8", errors="replace") as f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, bufsize=1
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")  
                f.write(line)

            rc = proc.wait()

        print(f"\n=== FULL LOG SAVED ===\n{log_path}\n")
        if rc != 0:
            raise SystemExit(rc)
    
    def import_cuBLAS_lib():
        import os, ctypes, pathlib
        import nvidia.cublas, nvidia.cuda_runtime

        cublas_dir = (pathlib.Path(nvidia.cublas.__file__).parent / "lib").resolve()
        cudart_dir = (pathlib.Path(nvidia.cuda_runtime.__file__).parent / "lib").resolve()

        os.environ["LD_LIBRARY_PATH"] = f"{cublas_dir}:{cudart_dir}:" + os.environ.get("LD_LIBRARY_PATH","")

        for name12, name in [("libcublas.so.12","libcublas.so"),
                            ("libcublasLt.so.12","libcublasLt.so")]:
            src = cublas_dir / name12
            dst = cublas_dir / name
            if src.exists() and not dst.exists():
                try: dst.symlink_to(name12)
                except FileExistsError: pass
                except PermissionError: pass 

        ctypes.CDLL(str(cudart_dir / "libcudart.so.12"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(cublas_dir / "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(cublas_dir / "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)

    if get_gpu_type():
        print("Running on H100 SXM")
    import_cuBLAS_lib()
    test_demo()