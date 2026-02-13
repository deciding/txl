from modal import Image, App, Volume
import pathlib
local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent
tutorial_path = root_dir / "python" / "txl" / "tutorials" / "mla"
requirements_file = root_dir / "requirements.txt"
#txl_wheel_file = local_dir / "txl-3.4.0-cp312-cp312-linux_x86_64.whl"
wheel_name = "txl-3.5.1-cp312-cp312-linux_x86_64.whl"
so_name = "cuda.cpython-312-x86_64-linux-gnu.so"
so1 = local_dir / "bins" / so_name
txl_wheel_file = local_dir / wheel_name

test_file_lib = tutorial_path / "lib.py"
# test_file = local_dir / "test_flash_mla_prefill.py"
test_file = tutorial_path / "test_flash_nsa_prefill.py"

flash_mla_dir = tutorial_path / "flash_mla"

app = App(name="txl-nsa-prefill")  # Note: this is optional since Modal 0.57
volume = Volume.from_name("txl-nsa-prefill-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

txl_image = (
    Image.debian_slim(python_version="3.12")
    #Image.from_dockerfile(path="./Dockerfile")
    .workdir("/workspace")
    .add_local_file(txl_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
    .run_commands( "ls .")
    .pip_install("torch==2.8.0") # for cuda.cpython is built on this version
    .pip_install_from_requirements(requirements_file) # local file not remote file
    .run_commands(
        f"pip install /workspace/{wheel_name}",
    )
    .add_local_file(test_file, remote_path="/workspace/test_txl.py", copy=False) # copy after image build, no need rebuild
    .add_local_file(test_file_lib, remote_path="/workspace/lib.py", copy=False)
    .add_local_dir(flash_mla_dir, remote_path="/workspace/flash_mla", copy=False)
    .add_local_file(so1, remote_path=f"/workspace/bins/{so_name}", copy=False)
)

# Example function that uses the image
@app.function(gpu="H100", image=txl_image, timeout=60*3,
		volumes={"/workspace/dump": volume})
def run_demo():
    import subprocess, sys, os, torch, time
    def get_gpu_type():
        sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        print(f"SM count: {sm}")
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
        from test_txl import main
        import torch
        main()

    get_gpu_type()
    test_demo()
