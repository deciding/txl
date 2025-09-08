import modal
import pathlib

app = modal.App("demo")

local_dir = pathlib.Path(__file__).parent.resolve()
volume = modal.Volume.from_name("flash-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(local_dir, remote_path="/workspace", copy=True) # copy the local code to the image
    .workdir("/workspace")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands("pip install /workspace/txl-3.4.0-cp312-cp312-linux_x86_64.whl") 
    # download form https://0x0.st/KXMu.whl to local, rename as txl-3.4.0-cp312-cp312-linux_x86_64.whl
)

@app.function(image=image, gpu="H100", timeout=60 * 60,
              volumes={"/workspace/dump": volume})  # mount the cloud volume to /workspace/dump in the container at runtime
def run_demo():
    import subprocess, sys, os, torch

    def test_torch():
        x= torch.randn(100, 100, device="cuda")
        x = x+x
        x[0].cpu()
    
    def get_gpu_type():
        try:
            result = subprocess.run(
                ['nvidia-smi', '-q'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            output = result.stdout
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
        env = os.environ.copy()
        env["TRITON_PRINT_AUTOTUNING"] = "1"
        env["TRITON_KERNEL_DUMP"] = "1"
        env["TRITON_DUMP_DIR"] = "/workspace/dump"
        env["TRITON_ALWAYS_COMPILE"] = "1"
        result = subprocess.run(
            [sys.executable, "demo-flash-attention.py"],
            capture_output=True,
            text=True,
            env=env,
        )
        print("=== STDOUT ===\n", result.stdout)
        print("=== STDERR ===\n", result.stderr)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    
    get_gpu_type()
    test_demo()
# download dump files to local
# modal volume get flash-dump [dump_flie_name] dump_local