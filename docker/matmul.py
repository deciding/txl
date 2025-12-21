from modal import Image, App, Volume
import pathlib
local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"

Use_TXL = False
app_name = 'txl' if Use_TXL else 'triton'


#txl
txl_wheel_file = local_dir / "txl-3.5.1-cp312-cp312-linux_x86_64.whl"

test_file = root_dir / "python" / "txl" / "tutorials" / "01-matmul.py"
ptx_file_name = "matmul_persistent_nows_tma_txl_bw_kernel"
ptx_file = local_dir / f"{ptx_file_name}.ptx"
signature_file = local_dir / f"{ptx_file_name}_signature.json"
json_file = local_dir / f"{ptx_file_name}.json"

# txl
app = App(name=f"{app_name}-matmul")  # Note: this is optional since Modal 0.57
volume = Volume.from_name(f"{app_name}-matmul-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

if Use_TXL:
    txl_image = (
        Image.debian_slim(python_version="3.12")
        # 2. Install tools required for CUDA repo
        .apt_install("wget", "curl", "gnupg")

        # 3. Add NVIDIA CUDA repository for Debian 12 (Bookworm)
        .run_commands(
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update"
        )

        # 4. Install CUDA debugging tools
        .apt_install(
            "cuda-toolkit-12-4",   # contains cuda-gdb
            # OR if you only want debugger:
            # "cuda-gdb",
        )

        .workdir("/workspace")
        # txl
        .add_local_file(txl_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
        .run_commands( "ls .")
        .pip_install_from_requirements(requirements_file) # local file not remote file
        # txl
        .run_commands(
            "pip install /workspace/txl-3.5.1-cp312-cp312-linux_x86_64.whl",
        )
        .env({
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib"
        })
        .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False) # copy after image build, no need rebuild
        .add_local_file(ptx_file, remote_path=f"/workspace/{ptx_file_name}.ptx", copy=False)
        .add_local_file(signature_file, remote_path=f"/workspace/{ptx_file_name}_signature.json", copy=False)
        .add_local_file(json_file, remote_path=f"/workspace/{ptx_file_name}.json", copy=False)
    )
else:
    txl_image = (
        Image.debian_slim(python_version="3.12")
        #Image.from_dockerfile(path="./Dockerfile")
        .workdir("/workspace")
        .run_commands( "ls .")
        .pip_install_from_requirements(requirements_file) # local file not remote file
        # triton
        .pip_install("triton==3.5.1")
        .env({
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib"
        })
        .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False) # copy after image build, no need rebuild
    )

# Example function that uses the image
#@app.function(gpu="H100", image=txl_image, timeout=60,
@app.function(gpu="B200", image=txl_image, timeout=300,
		volumes={"/workspace/dump": volume})
def test_flash_attention():

    def get_gpu_type():
        import subprocess
    
        try:
            result = subprocess.run(['find', '/', '-name', 'libcublas.so*'], capture_output=True, text=True, check=True)
            output = result.stdout
            print(output)
            result = subprocess.run(['find', '/', '-name', 'cuda-gdb'], capture_output=True, text=True, check=True)
            output = result.stdout
            print(output)
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    #if 'H100' in line and 'HBM3' in line:
                    if 'B200' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return False

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


    if not get_gpu_type():
        return
    import_cuBLAS_lib()

    from test_txl import test_matmul
    test_matmul(None, "0")
    #test_matmul("/workspace/dump", "b3")
    #test_matmul(None, "b4")
    #import subprocess
    #p = subprocess.Popen(
    #        ["/usr/local/cuda-12.4/bin/cuda-gdb", "-ex", "run", "--args", "python", "/workspace/test_txl.py"],
    #        #stdout=subprocess.PIPE, stderr=subprocess.PIPE
    #)
    #p.wait()


