from modal import Image, App, Volume
import pathlib
import os

local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"

# Get dump directory name from environment (passed by modal_tests.sh)
DUMP_DIR = os.environ.get("DUMP_DIR", "default")

# Get wheel filename from environment or find in dist directory
# This runs on HOST when modal run is executed
txl_wheel_name = os.environ.get("TXL_WHEEL_NAME")
if not txl_wheel_name:
    dist_dir = root_dir / "thirdparty" / "triton" / "dist"
    # Search for both teraxlang-*.whl and txl-*.whl
    wheel_files = list(dist_dir.glob("teraxlang-*.whl")) + list(
        dist_dir.glob("txl-*.whl")
    )
    if wheel_files:
        # Prefer linux_x86_64 wheel
        txl_wheel = next(
            (f for f in wheel_files if "linux_x86_64" in f.name), wheel_files[0]
        )
        txl_wheel_name = txl_wheel.name
        print(f"Using wheel: {txl_wheel_name}")
    else:
        txl_wheel_name = "teraxlang-3.5.1-cp312-cp312-linux_x86_64.whl"  # fallback

print(f"Dump directory: {DUMP_DIR}")

test_file = root_dir / "python" / "teraxlang" / "tutorials" / "02-flash-attention.py"

app = App(name="txl-mac")  # Note: this is optional since Modal 0.57
volume = Volume.from_name(
    "txl-dump", create_if_missing=True
)  # create a cloud volume to store compiled dump files

# Find wheel file path on host
txl_wheel_file = root_dir / "thirdparty" / "triton" / "dist" / txl_wheel_name

txl_image = (
    Image.debian_slim(python_version="3.12")
    # Image.from_dockerfile(path="./Dockerfile")
    .workdir("/workspace")
    .add_local_file(
        txl_wheel_file, remote_path="/workspace/", copy=True
    )  # copy the local code to the image
    .run_commands("ls .")
    .pip_install_from_requirements(requirements_file)  # local file not remote file
    .run_commands(
        f"pip install /workspace/{txl_wheel_name}",
    )
    .add_local_file(
        test_file, remote_path="/workspace/test_txl.py", copy=False
    )  # copy after image build, no need rebuild
    # .add_local_file(ptx_file, remote_path="/workspace/_attn_fwd_ws_tma_txl_tawa.ptx", copy=False) # copy after image build, no need rebuild
    # .add_local_file(signature_file, remote_path="/workspace/_attn_fwd_ws_tma_txl_tawa_signature.json", copy=False) # copy after image build, no need rebuild
    # .add_local_file(json_file, remote_path="/workspace/_attn_fwd_ws_tma_txl_tawa.json", copy=False) # copy after image build, no need rebuild
)


# Example function that uses the image
@app.function(
    gpu="H100", image=txl_image, timeout=60, volumes={"/workspace/dump": volume}
)
def test_flash_attention(dump_dir: str = "default"):
    dump_path = f"/workspace/dump/{dump_dir}"

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
                    if "H100" in line and "HBM3" in line:
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

    from test_txl import run_test

    # Create dump directory and run test
    os.makedirs(dump_path, exist_ok=True)
    print(f"Dump path: {dump_path}")
    run_test("hopper_txl_ws_fa3", dump_path)


# Local entrypoint - reads DUMP_DIR from environment and passes to function
@app.local_entrypoint()
def main():
    dump_dir = os.environ.get("DUMP_DIR", "default")
    print(f"Running test with dump_dir: {dump_dir}")
    test_flash_attention.remote(dump_dir)
