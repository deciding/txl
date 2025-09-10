from modal import Image, App, Volume
import pathlib
local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"
txl_wheel_file = local_dir / "txl-3.4.0-cp312-cp312-linux_x86_64.whl"

test_file = root_dir / "python" / "txl" / "tutorials" / "09-persistent-matmul.py"

app = App(name="txl")  # Note: this is optional since Modal 0.57
volume = Volume.from_name("txl-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

txl_image = (
    Image.debian_slim(python_version="3.12")
    #Image.from_dockerfile(path="./Dockerfile")
    .workdir("/workspace")
    .add_local_file(txl_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
    .run_commands( "ls .")
    .pip_install_from_requirements(requirements_file) # local file not remote file
    .run_commands(
        "pip install /workspace/txl-3.4.0-cp312-cp312-linux_x86_64.whl",
    )
    .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False) # copy after image build, no need rebuild
)

# Example function that uses the image
@app.function(gpu="B200", image=txl_image, timeout=60,
		volumes={"/workspace/dump": volume})
def run_demo():
    import subprocess, sys, os, torch
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
        env = os.environ.copy()
        env["TRITON_PRINT_AUTOTUNING"] = "1"
        env["TRITON_KERNEL_DUMP"] = "1"
        env["TRITON_DUMP_DIR"] = "/workspace/dump"
        env["TRITON_ALWAYS_COMPILE"] = "1"
        result = subprocess.run(
            [sys.executable, "/workspace/test_txl.py"],
            capture_output=True,
            text=True,
            env=env,
        )
        print("=== STDOUT ===\n", result.stdout)
        print("=== STDERR ===\n", result.stderr)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    
    test_demo()