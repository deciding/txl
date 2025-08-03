from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.semantic import (
    device_print,
    to_tensor,
    validate_descriptor_block,
    _convert_to_ir_values,
    broadcast_impl_shape,
    cast,
)
from typing import Optional, Tuple

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

def _str_to_padding_optionx(padding_option):
    padding = None  # default
    if padding_option:
        if padding_option == "zero":
            padding = ir.PADDING_OPTIONX.PAD_ZERO
        elif padding_option == "nan":
            padding = ir.PADDING_OPTIONX.PAD_NAN
        else:
            raise ValueError(f"Padding option {padding_option} not supported")
    return padding


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

def is_warpgroup(ids, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_is_warpgroup(ids), tl.int1)

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

def tma_gather(value, desc, x_offsets, y_offset, cache_modifier: str, eviction_policy: str,
                      builder: ir.builder) -> tl.tensor:
    assert isinstance(desc, tl._experimental_tensor_descriptor_base)
    assert cache_modifier == "", "cache modifier is not supported yet"
    assert eviction_policy == "", "eviction policy is not supported yet"

    # Validate descriptor.
    assert len(desc.block_shape) == 2, f"descriptor must be 2D, but got {desc.block_shape}"
    assert desc.block_shape[0] == 1, f"descriptor block must have 1 row, but got {desc.block_shape}"

    # Validate offsets.
    assert len(x_offsets.shape) == 1, f"x offsets must be 1D, but got {x_offsets.shape}"

    # Validate minimum block size.
    assert x_offsets.shape[0] >= 8, f"descriptor gather must have at least 8 rows, but got {x_offsets.shape}"
    dtype = desc.dtype
    min_cols = 32 // dtype.primitive_bitwidth * 8
    assert desc.block_shape[
        1] >= min_cols, f"descriptor gather of {dtype} must have at least {min_cols} columns, but got {desc.block_shape[1]}"

    type = tl.block_type(desc.dtype, [x_offsets.shape[0], desc.block_shape[1]])
    y_offset = _convert_to_ir_values(builder, (y_offset, ), require_i64=False)[0]
    x = builder.create_tma_gather(value.handle, desc.handle, x_offsets.handle, y_offset, type.to_ir(builder))
    return tl.tensor(x, tl.void)

def dot_wait(pendings:int, builder: ir.builder) -> tl.tensor:
    x = builder.create_dot_wait(pendings)
    return tl.tensor(x, tl.void)

def bar_arrive(bar: tl.tensor, num_threads: tl.tensor, builder: ir.builder) -> tl.tensor:
    x = builder.create_bar_arrive(bar.handle, num_threads.handle)
    return tl.tensor(x, tl.void)

def bar_wait(bar: tl.tensor, num_threads: tl.tensor, builder: ir.builder) -> tl.tensor:
    x = builder.create_bar_wait(bar.handle, num_threads.handle)
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

def mbar_arrive(mbar: tl.tensor, pred:tl.tensor, track_async_op:bool, tx_cnt:int, builder: ir.builder) -> tl.tensor:
    x = builder.create_mbar_arrive(mbar.handle, pred.handle, track_async_op, tx_cnt)
    return tl.tensor(x, tl.void)

def _async_load_legacy(mem, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

    # Check `mask`, `other`, `boundary_check`, and `padding` arguments
    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                         "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                         "use block pointers (defined by `make_block_ptr`) instead")

    # For a pointer of scalar, check the type of `mask` and `other`
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
        if other and other.type.is_block():
            raise ValueError("Other argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `other` into the same shape as `ptr`
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other is not None:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    # Get `pointer_type<elt_ty>` and `elt_ty`
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast `other` into `elt_ty` type
    if other is not None:
        other = cast(other, elt_ty, builder)

    # Create loaded result type `dst_ty`
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = elt_ty

    # Build IR
    if mask is None:
        ret = tl.tensor(builder.create_async_load(mem.handle, ptr.handle, cache, eviction, is_volatile), dst_ty)
    else:
        ret = tl.tensor(
            builder.create_masked_async_load(mem.handle, ptr.handle, mask.handle, other.handle if other else None, cache, eviction,
                                       is_volatile), dst_ty)
    if is_bool:
        ret = cast(ret, tl.int1, builder)
    return ret

# Currently don't have RewriteTensorPointer, no block pointer boundary check
# also no Combine, PlanCTA and AxisInfo logics
def async_load(mem: tl.tensor, ptr: tl.tensor, mask: Optional[tl.tensor], other: Optional[tl.tensor],
        boundary_check: Tuple, padding_option: str,
        cache_modifier: str, eviction_policy: str, is_volatile: bool,
         builder: ir.builder) -> tl.tensor:
    # Cache, eviction and padding options
    cache = _str_to_load_cache_modifierx(cache_modifier)
    eviction = _str_to_eviction_policyx(eviction_policy)

    assert not (ptr.type.is_ptr() and ptr.type.element_ty.is_block()), 'block pointer not supported for async_load for now'

    # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    return _async_load_legacy(mem, ptr, mask, other, boundary_check, None, cache, eviction, is_volatile, builder)

def async_load_wait(async_token: tl.tensor, num:int, builder: ir.builder) -> tl.tensor:
    x = builder.create_async_load_wait(async_token.handle, num)
    return tl.tensor(x, tl.void)
