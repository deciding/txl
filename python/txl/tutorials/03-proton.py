import torch

import triton
import triton.language as tl
import triton.profiler.language as pl
import triton.profiler as proton
import pathlib
import os

from typing import NamedTuple

import txl
from txl.language.semantic import TXLSemantic

DEVICE = triton.runtime.driver.active.get_active_torch_device()

pl.enable_semantic("triton")
pl.enable_semantic_obj(TXLSemantic)


def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    BLOCK_SIZE = args["BLOCK_SIZE"]
    return {"name": f"add_{BLOCK_SIZE}"}


@txl.autotune(
        configs = [
            txl.Config(
                {},
                num_warps=4,
                num_warpgroups=2,
                #ir_override='dump/test_proton/inner_dir/add_kernel.ttgir',
            )
        ],
        key=["BLOCK_SIZE"]
)
#@txl.jit(launch_metadata=metadata_fn, diff_mode='llir', log_dir='dump')
#@txl.jit(launch_metadata=metadata_fn, diff_mode='llir')
@txl.jit(launch_metadata=metadata_fn)
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    with pl.scope("kernel"):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        if txl.is_warpgroup([0, 1]):
            txl.reg_dealloc(40)
            with pl.scope("wg01"):
                if txl.is_warpgroup([0]):
                    offsets = block_start + tl.arange(0, BLOCK_SIZE // 2)
                else:
                    offsets = block_start + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE)
                mask = offsets < n_elements

                for i in range(3):
                    with pl.scope("load_ops"):
                        with pl.scope("load_x"):
                            x = tl.load(x_ptr + offsets, mask=mask)
                        with pl.scope("load_y"):
                            y = tl.load(y_ptr + offsets, mask=mask)

                    if txl.is_warpgroup([0]):
                        with pl.scope("store_ops0"):
                            tid = txl.tid(0)
                            output = x + y + tid
                    else:
                        output = x + y
                    tl.store(output_ptr + offsets, output, mask=mask)

@txl.autotune(
        configs = [
            txl.Config(
                {},
                num_warps=4,
                num_warpgroups=2,
                #ir_override='dump/test_proton/inner_dir/add_kernel.ttgir',
            )
        ],
        key=["BLOCK_SIZE"]
)
@txl.jit(launch_metadata=metadata_fn)
def add_kernel1(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    with pl.scope("kernel"):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        if txl.is_warpgroup([0]):
            with pl.scope("wg0"):
                offsets = block_start + tl.arange(0, BLOCK_SIZE // 2)
                mask = offsets < n_elements

                for i in range(3):
                    with pl.scope("load_ops"):
                        with pl.scope("load_x"):
                            x = tl.load(x_ptr + offsets, mask=mask)
                        with pl.scope("load_y"):
                            y = tl.load(y_ptr + offsets, mask=mask)

                    with pl.scope("store_ops"):
                        tid = txl.tid(0)
                        output = x + y + tid
                    tl.store(output_ptr + offsets, output, mask=mask)
        else:
            with pl.scope("wg1"):
                offsets = block_start + tl.arange(0, BLOCK_SIZE // 2)
                mask = offsets < n_elements

                for i in range(3):
                    with pl.scope("load_ops"):
                        with pl.scope("load_x"):
                            x = tl.load(x_ptr + offsets, mask=mask)
                        with pl.scope("load_y"):
                            y = tl.load(y_ptr + offsets, mask=mask)

                    with pl.scope("store_ops"):
                        tid = txl.tid(0)
                        output = x + y + tid
                    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    tmp_path = pathlib.Path(os.getcwd())

    temp_file = tmp_path / "vector-add.json"

    #proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode="default:granularity=warp", data="trace")
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", data="trace")

    #add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=4, num_warpgroups=2)
    #add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    proton.finalize()

    return output


torch.manual_seed(0)
size = 2048
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
