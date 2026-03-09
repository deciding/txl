from modal import Image, App, Volume
import pathlib
import os

local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
requirements_file = root_dir / "requirements.txt"

DUMP_DIR = os.environ.get("DUMP_DIR", "default")
TRITON_LLVM_DEBUG_ONLY = os.environ.get("TRITON_LLVM_DEBUG_ONLY", "")

txl_wheel_name = os.environ.get("TXL_WHEEL_NAME")
main_cmd = None
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
        use_pip = True
        print("Using Pip")
        main_cmd = "pip install teraxlang"

if txl_wheel_name:
    txl_wheel_file = root_dir / "thirdparty" / "triton" / "dist" / txl_wheel_name
else:
    txl_wheel_file = None

print(f"Dump directory: {DUMP_DIR}")

tutorial_path = root_dir / "python" / "teraxlang" / "tutorials" / "mla"
test_file = tutorial_path / "test_flash_nsa_prefill.py"
test_file_lib = tutorial_path / "lib.py"
flash_mla_dir = tutorial_path / "flash_mla"
bins_dir = local_dir / "bins"

app = App(name="txl-nsa-prefill")
volume = Volume.from_name("txl-dump", create_if_missing=True)

txl_image = Image.debian_slim(python_version="3.12").workdir("/workspace")
if txl_wheel_file:
    txl_image = txl_image.add_local_file(
        txl_wheel_file, remote_path="/workspace/", copy=True
    )
txl_image = (
    txl_image.run_commands("ls .")
    .pip_install("torch==2.8.0")
    .pip_install_from_requirements(requirements_file)
    .run_commands(main_cmd)
    .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False)
    .add_local_file(test_file_lib, remote_path="/workspace/lib.py", copy=False)
    .add_local_dir(flash_mla_dir, remote_path="/workspace/flash_mla", copy=False)
    .add_local_dir(bins_dir, remote_path="/workspace/bins", copy=False)
)


@app.function(
    gpu="H100", image=txl_image, timeout=60, volumes={"/workspace/dump": volume}
)
def test_nsa_prefill(dump_dir: str = "default", triton_debug: str = ""):
    # Set debug env var if provided
    if triton_debug:
        os.environ["TRITON_LLVM_DEBUG_ONLY"] = triton_debug
        print(f"Set TRITON_LLVM_DEBUG_ONLY={triton_debug}")

    import subprocess
    import torch

    def get_gpu_type():
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q"], capture_output=True, text=True, check=True
            )
            for line in result.stdout.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if "H100" in line and "HBM3" in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found.")
        return False

    if not get_gpu_type():
        return

    dump_path = f"/workspace/dump/{dump_dir}"
    os.makedirs(dump_path, exist_ok=True)
    print(f"Dump path: {dump_path}")

    from test_txl import main

    main(dump_dir=dump_path)


@app.local_entrypoint()
def main():
    dump_dir = os.environ.get("DUMP_DIR", "default")
    triton_debug = os.environ.get("TRITON_LLVM_DEBUG_ONLY", "")
    print(f"Running test with dump_dir: {dump_dir}")
    if triton_debug:
        print(f"Running with TRITON_LLVM_DEBUG_ONLY={triton_debug}")
    test_nsa_prefill.remote(dump_dir, triton_debug)
