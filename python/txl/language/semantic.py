from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.semantic import (
    device_print,
    to_tensor,
    load,
    validate_descriptor_block,
    _convert_to_ir_values,
)

def _str_to_load_cache_modifierx(cache_modifier):
    cache = ir.CACHE_MODIFIERX.NONE  # default
    if cache_modifier:
        if cache_modifier == ".ca":
            cache = ir.CACHE_MODIFIERX.CA
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIERX.CG
        elif cache_modifier == ".cv":
            cache = ir.CACHE_MODIFIERX.CV
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")
    return cache

def _str_to_eviction_policyx(eviction_policy):
    eviction = ir.EVICTION_POLICYX.NORMAL  # default
    if eviction_policy:
        if eviction_policy == "evict_last":
            eviction = ir.EVICTION_POLICYX.EVICT_LAST
        elif eviction_policy == "evict_first":
            eviction = ir.EVICTION_POLICYX.EVICT_FIRST
        else:
            raise ValueError(f"Eviction policy {eviction_policy} not supported")
    return eviction

def threadIdx(axis:int, builder: ir.builder) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"thread index axis must be 0, 1, or 2 but got {axis}")
    handle = builder.create_get_threadidx(axis)
    return tl.tensor(builder.create_index_cast(handle, tl.int32.to_ir(builder)), tl.int32)

def blockDim(axis:int, builder: ir.builder) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"block dim axis must be 0, 1, or 2 but got {axis}")
    handle = builder.create_get_blockdim(axis)
    return tl.tensor(builder.create_index_cast(handle, tl.int32.to_ir(builder)), tl.int32)

def blockIdx(axis:int, builder: ir.builder) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"block index axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)

def gridDim(axis:int, builder: ir.builder) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"grid dim axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(builder.create_get_num_programs(axis), tl.int32)

def warp_id(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_canonical_wrap_id(), tl.int32)

def warpgroup_id(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_canonical_wrapgroup_id(), tl.int32)

def reg_alloc(count:int, builder: ir.builder):
    builder.create_reg_alloc(count)

def reg_dealloc(count:int, builder: ir.builder):
    builder.create_reg_dealloc(count)

def smem_alloc(shape, dtype: tl.dtype, builder: ir.builder, num_stages:int=1, mutable:bool=True) -> tl.tensor:
    block_type = tl.block_type(dtype, shape)
    dtype = dtype.to_ir(builder)
    return tl.tensor(
        builder.create_smem_alloc(shape, dtype, num_stages, mutable), block_type)

def mbar_alloc(arr_count: int, builder: ir.builder, num_stages:int=1) -> tl.tensor:
    block_type = tl.block_type(tl.int64, [1])
    return tl.tensor(
        builder.create_mbar_alloc(arr_count, num_stages), block_type)

def tma_load(value: tl.tensor, desc: tl._experimental_tensor_descriptor_base, offsets,
                mbar: tl.tensor,
                cache_modifier: str, eviction_policy: str,
                builder: ir.builder) -> tl.tensor:
    assert isinstance(desc, tl._experimental_tensor_descriptor_base)
    validate_descriptor_block(desc.block_shape, desc.dtype)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)
    x = builder.create_tma_load(value.handle, desc.handle, offsets, mbar.handle, _str_to_load_cache_modifierx(cache_modifier),
                                       _str_to_eviction_policyx(eviction_policy))
    return tl.tensor(x, tl.void)

def dot_wait(pendings:int, builder: ir.builder) -> tl.tensor:
    x = builder.create_dot_wait(pendings)
    return tl.tensor(x, tl.void)

def bar_arrive(bar: tl.tensor, num_threads: tl.tensor, builder: ir.builder) -> tl.tensor:
    x = builder.create_bar_arrive(bar, num_threads)
    return tl.tensor(x, tl.void)

def bar_wait(bar: tl.tensor, num_threads: tl.tensor, builder: ir.builder) -> tl.tensor:
    x = builder.create_bar_wait(bar, num_threads)
    return tl.tensor(x, tl.void)

def get_buffer(src: tl.tensor, index: tl.tensor, builder: ir.builder) -> tl.tensor:
    # TODO: check
    x = builder.create_get_buffer(src.handle, index.handle)
    return tl.tensor(x, src.type)

def mbar_expect(mbar: tl.tensor, size_in_bytes:int, pred: tl.tensor, builder: ir.builder) -> tl.tensor:
    # TODO: handle pred is None, then should be const True
    x = builder.create_mbar_expect(mbar.handle, pred.handle, size_in_bytes)
    return tl.tensor(x, tl.void)

def mbar_wait(mbar: tl.tensor, phase:tl.tensor, builder: ir.builder) -> tl.tensor:
    x = builder.create_mbar_wait(mbar.handle, phase.handle)
    return tl.tensor(x, tl.void)
