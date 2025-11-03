import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import txl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

"""
Full smem
"""
@txl.jit()
def txl_kernel(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    s_buf = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)
    rid = txl.cta_rank()
    tid = txl.tid(0)
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32)
    if rid == 1:
        x += 1
    txl.smem_store(s, x)
    layout :tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [1, 1], [1, 0], ctas_per_cga=[2, 1], cta_split_num=[2, 1], cta_order=[1, 0])
    y = txl.smem_load(s, layout)
    desc_o.store([0, 0], y)

"""
Normal smem
"""
@txl.jit()
def txl_kernel0(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    s_buf = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)
    tid = txl.tid(0)
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32)
    x += 1
    txl.smem_store(s, x)
    txl.tma_store(s, desc_o, [0, 0])

"""
half smem
"""
@txl.jit()
#@txl.jit(diff_mode='llir')
#@txl.jit(src_file='dump/cluster/ACY7IPOZL6RTOBW4DMV5WZFDHLF67GR5533DFTBNU2EQWKN4V2SA/txl_kernel1.ptx')
def txl_kernel1(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    s_buf = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)
    #s_buf = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)
    rid = txl.cta_rank()
    tid = txl.tid(0)
    #x = tl.zeros([BLOCK_SIZE_M//2, BLOCK_SIZE_N], tl.float32)
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32)
    if rid == 0:
        x += 1
    else:
        x += 2
    txl.smem_store(s, x)
    #layout :tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [1, 1], [1, 0], ctas_per_cga=[2, 1], cta_split_num=[2, 1], cta_order=[1, 0])
    #y = txl.smem_load(s, layout)
    #if tid == 0:
    #    txl.print('y', y)
    #    txl.print('rid', rid)
    txl.tma_store(s, desc_o, [0, 0])
    #if rid == 0:
    #    txl.tma_store(s, desc_o, [0, 0])
    #else:
    #    txl.tma_store(s, desc_o, [BLOCK_SIZE_M//2, 0])

"""
two smem
"""
@txl.jit()
#@txl.jit(diff_mode='llir', log_dir='dump/cluster')
#@txl.jit(src_file='dump/cluster/ACY7IPOZL6RTOBW4DMV5WZFDHLF67GR5533DFTBNU2EQWKN4V2SA/txl_kernel1.ptx')
def txl_kernel2(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    s_buf = txl.smem_alloc([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)

    rid = txl.cta_rank()
    tid = txl.tid(0)
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32)
    if rid == 0:
        x += 1.0
        txl.smem_store(s, x, 1)
    else:
        x += 2.0
        txl.smem_store(s, x, 0)
    #txl.smem_store(s, x)
    #layout :tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [1, 1], [1, 0], ctas_per_cga=[2, 1], cta_split_num=[2, 1], cta_order=[1, 0])
    #y = txl.smem_load(s, layout)
    #if tid == 0:
    #    txl.print('y', y)
    #    txl.print('rid', rid)
    txl.tma_store(s, desc_o, [0, 0])


def test_txl():
    dtype = torch.float32
    BLOCK_SIZE_M = 16 * 2 * 4
    BLOCK_SIZE_N = 64

    # full
    dummy_block = [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # half
    #dummy_block = [BLOCK_SIZE_M//2, BLOCK_SIZE_N]

    q = torch.randn((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype, device=DEVICE)
    o = torch.empty_like(q)

    desc_q = TensorDescriptor(q, shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], strides=[BLOCK_SIZE_N, 1], block_shape=dummy_block)
    #desc_o = TensorDescriptor(o, shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], strides=[BLOCK_SIZE_N, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], strides=[BLOCK_SIZE_N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    grid = lambda meta: (1,)
    txl_kernel2[grid](desc_q, desc_o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=4, num_ctas=2)
    #txl_kernel1[grid](desc_q, desc_o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=4, num_ctas=2)
    #txl_kernel[grid](desc_q, desc_o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=1)

    print(q.shape)
    print(q)
    print(o.shape)
    print(o)



"""
frag_smem_store
"""
@txl.jit()
def txl_kernel3(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    layout :tl.constexpr = txl.BlockedLayout([1, 1], [1, 32], [1, 2], [1, 0], ctas_per_cga=[2, 1], cta_split_num=[2, 1], cta_order=[1, 0])
    s_buf = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)

    rid = txl.cta_rank()
    tid = txl.tid(0)
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32)
    if rid == 0:
        x += 1.0
        txl.frag_smem_store(s, x, layout, 1)
    else:
        x += 2.0
        txl.frag_smem_store(s, x, layout, 0)
    txl.tma_store(s, desc_o, [0, 0])

"""
frag_smem_store
"""
@txl.jit()
def txl_kernel31(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    layout :tl.constexpr = txl.BlockedLayout([1, 1], [1, 32], [1, 2], [1, 0], ctas_per_cga=[2, 1], cta_split_num=[2, 1], cta_order=[1, 0])
    s_buf = txl.smem_alloc([BLOCK_SIZE_M//2, BLOCK_SIZE_N], tl.float32); s = txl.get_buffer(s_buf, 0)

    rid = txl.cta_rank()
    tid = txl.tid(0)
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32)
    if rid == 0:
        x += 1.0
        txl.frag_smem_store(s, x, layout, 1)
    else:
        x += 2.0
        txl.frag_smem_store(s, x, layout, 0)
    txl.tma_store(s, desc_o, [0, 0])

def test_txl3():
    dtype = torch.float32
    BLOCK_SIZE_M = 16 * 2 * 4
    BLOCK_SIZE_N = 64

    # full
    dummy_block = [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # half
    #dummy_block = [BLOCK_SIZE_M//2, BLOCK_SIZE_N]

    q = torch.randn((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype, device=DEVICE)
    o = torch.zeros((BLOCK_SIZE_M//2, BLOCK_SIZE_N), dtype=dtype, device=DEVICE)

    desc_q = TensorDescriptor(q, shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], strides=[BLOCK_SIZE_N, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[BLOCK_SIZE_M//2, BLOCK_SIZE_N], strides=[BLOCK_SIZE_N, 1], block_shape=[BLOCK_SIZE_M//2, BLOCK_SIZE_N])

    grid = lambda meta: (1,)
    txl_kernel3[grid](desc_q, desc_o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=4, num_ctas=2)

    print(q.shape)
    print(q)
    print(o.shape)
    print(o)

if __name__ == '__main__':
    dump_dir = None
    dump_dir='dump/cluster/'

    from triton import knobs
    import os
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-legalize"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    knobs.runtime.override_arch='sm90'
    knobs.autotuning.print=True
    knobs.compilation.always_compile=True

    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir

    #test_txl()
    test_txl3()
