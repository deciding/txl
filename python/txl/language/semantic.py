from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.semantic import (
    device_print,
    to_tensor,
    load,
)

def threadIdx(builder: ir.builder, axis:int=0) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"thread index axis must be 0, 1, or 2 but got {axis}")
    handle = builder.create_get_threadidx(axis)
    return tl.tensor(builder.create_index_cast(handle, tl.int32.to_ir(builder)), tl.int32)

def blockDim(builder: ir.builder, axis:int=0) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"block dim axis must be 0, 1, or 2 but got {axis}")
    handle = builder.create_get_blockdim(axis)
    return tl.tensor(builder.create_index_cast(handle, tl.int32.to_ir(builder)), tl.int32)

def blockIdx(builder: ir.builder, axis:int=0) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"block index axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)

def gridDim(builder: ir.builder, axis:int=0) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"grid dim axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(builder.create_get_num_programs(axis), tl.int32)

def warp_id(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_canonical_wrap_id(), tl.int32)

def warpgroup_id(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_canonical_wrapgroup_id(), tl.int32)

def reg_alloc(builder: ir.builder, count:int):
    builder.create_reg_alloc(count)

def reg_dealloc(builder: ir.builder, count:int):
    builder.create_reg_dealloc(count)

def smem_alloc(builder: ir.builder, shape, dtype: tl.dtype, mutable:bool=False) -> tl.tensor:
    block_type = tl.block_type(dtype, shape)
    dtype = dtype.to_ir(builder)
    return tl.tensor(
        builder.create_smem_alloc(shape, dtype, mutable), block_type)
