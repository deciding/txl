import torch

import triton
import triton.language as tl
import txl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

co = 64
BLOCK_SIZE_O = 64

#@txl.jit
#@txl.jit(diff_mode='llir', log_dir='log')
@txl.jit(diff_mode='ttgir', log_dir='log')
#@txl.jit(diff_mode='ttir', log_dir='log')
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cn, stride_co,
        stride_dm, stride_do,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        BLOCK_SIZE_O: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    #pid = tl.program_id(axis=0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_m = (tl.arange(0, BLOCK_SIZE_M))
    offs_n = (tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + stride_cn * offs_n[:, None] + stride_co * offs_o[None, :]
    d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_do * offs_o[None, :]

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    #accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Load the next block of A and B, generate a mask by checking the K dimension.
    # If it is out of bounds, set it to 0.
    #a = txl.load(a_ptrs, 'smem')
    a = txl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.load(c_ptrs)
    a*=3
    a+=6
    # We accumulate along the K dimension.
    ab = tl.dot(a, b)
    ab = ab.to(tl.float16)
    #ab *= 3

    #ab = accumulator.to(tl.float16)


    d = tl.dot(ab, c)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    tl.store(d_ptrs, d)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, c, O):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    M, N = c.shape
    # Allocates output.
    d = torch.empty((M, O), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (1, )
    matmul_kernel[grid](
        a, b, c, d, #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        d.stride(0), d.stride(1),  #
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, BLOCK_SIZE_O=BLOCK_SIZE_O
    )
    return d


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
a = torch.randn((128, 128), device=DEVICE, dtype=torch.float16)
b = torch.randn((128, 128), device=DEVICE, dtype=torch.float16)
c = torch.randn((128, co), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b, c, O=co)
#torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
#print(f"torch_output_with_fp16_inputs={torch_output}")
