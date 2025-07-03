import torch

import triton
import triton.language as tl

import txl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit()
#@txl.jit()
def print_kernel(
        BLOCK_SIZE: tl.constexpr,
               ):
    if txl.tid(0) == 0:
        txl.print('hello t0')
    #if txl.warp_id() == 0:
    #    txl.print('hello w0', txl.tidx.x())
    #if txl.warpgroup_id() == 0:
    #    txl.print('hello wg0', txl.tidx.x())


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def test_print():
    grid = lambda meta: (triton.cdiv(4096, meta['BLOCK_SIZE']), )
    print_kernel[grid](BLOCK_SIZE=1024)

test_print()
