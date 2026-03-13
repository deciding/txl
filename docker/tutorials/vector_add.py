"""
Modal script to run TXL vector add benchmark on H100 GPU.
"""

from datetime import datetime
from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent.parent
GPU_model = "H100"

APP_NAME = "txl-vector-add"
VOLUME_NAME = "txl-dump"
DUMP_VOL = "/workspace/dump"
DUMP_DIR = APP_NAME + "".join(str(datetime.now()).replace(":", ".").split())

app = App(name=APP_NAME)

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

txl_image = (
    Image.debian_slim(python_version="3.12")
    .workdir("/workspace")
    .pip_install("torch", "pytest")
    .pip_install("teraxlang==3.5.1.dev2")
    .add_local_dir(root_dir / "tutorials", remote_path="/workspace/tutorials")
)


@app.function(
    gpu=GPU_model,
    image=txl_image,
    timeout=600,
    volumes={DUMP_VOL: volume},
)
def run_vector_add():
    import torch
    import sys
    import triton
    import triton.language as tl
    import teraxlang as txl
    from teraxlang.tools.build_binding_view import generate_htmls

    # TeraXLang Settings
    from triton import knobs

    knobs.autotuning.print = True
    knobs.compilation.always_compile = True
    if DUMP_DIR:
        knobs.compilation.dump_ir = True
        knobs.cache.dump_dir = f"{DUMP_VOL}/{DUMP_DIR}"

    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    SIZE = 1024
    BLOCK_SIZE = 1024

    @txl.jit()
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        bid = txl.bid(0)
        block_start = bid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        s = tl.sum(output)
        output -= s
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x, device=DEVICE)
        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

    x = torch.rand(SIZE, device=DEVICE, dtype=torch.float32)
    y = torch.rand(SIZE, device=DEVICE, dtype=torch.float32)

    n_elements = SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    output = add(x, y)
    torch.cuda.synchronize()

    result = x + y
    s = result.sum()
    result -= s

    if torch.allclose(output, result):
        print("SUCCESS: Output matches expected result!")
    else:
        print("ERROR: Output does not match expected result!")
        max_diff = (output - result).abs().max()
        print(f"Max diff: {max_diff}")

    # Generate HTML viewers for IR files
    dump_path = f"{DUMP_VOL}/{DUMP_DIR}"
    print(f"\n=== Generating HTML viewers for IR files ===")
    print(f"IR dump directory: {dump_path}")

    # Python source file is mounted at /workspace/tutorials/vector_add.py
    py_file = "/workspace/tutorials/vector_add.py"

    # Generate HTML viewers for all IR files in the dump directory
    generate_htmls(dump_path, py_file, verbose=True)

    print(f"\nHTML viewers generated!")
    print(f"to download and view: modal volume get {VOLUME_NAME} {DUMP_DIR}")


@app.local_entrypoint()
def main():
    run_vector_add.remote()
