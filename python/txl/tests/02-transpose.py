import triton
import triton.language as tl
import torch

import txl


@txl.jit(diff_mode='ttgir')
def transpose_kernel(
    A_ptr, B_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)  # block row
    pid_n = tl.program_id(1)  # block col

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load a block from A into shared memory (local cache)
    tile = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask_a)

    # Transpose the tile in shared memory
    tile = tl.trans(tile)

    # Compute output offsets (note the transpose: M <-> N)
    offs_mt = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_nt = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_b = (offs_mt[:, None] < N) & (offs_nt[None, :] < M)

    # Store the transposed tile into B
    tl.store(B_ptr + offs_mt[:, None] * stride_bm + offs_nt[None, :] * stride_bn, tile, mask=mask_b)

M, N = 1024, 2048
BLOCK_SIZE = 32

A = torch.randn((M, N), device='cuda')
B = torch.empty((N, M), device='cuda')

grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']), triton.cdiv(N, META['BLOCK_SIZE']))

transpose_kernel[grid](
    A, B,
    M, N,
    A.stride(0), A.stride(1),
    B.stride(0), B.stride(1),
    BLOCK_SIZE=BLOCK_SIZE,
)
