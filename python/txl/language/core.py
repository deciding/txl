from triton.language.core import builtin
from . import semantic
from .utils import _constexpr_to_value, _apply_binary_method

@builtin
def tid(axis, _builder=None):
    return semantic.threadIdx(_builder, axis)

@builtin
def tdim(axis, _builder=None):
    return semantic.blockDim(_builder, axis)

@builtin
def bid(axis, _builder=None):
    return semantic.blockIdx(_builder, axis)

@builtin
def bdim(axis, _builder=None):
    return semantic.gridDim(_builder, axis)

@builtin
def thread0(_builder=None):
    is_bidx_x = bid(0, _builder=_builder)
    is_bidx_y = bid(1, _builder=_builder)
    is_bidx_z = bid(2, _builder=_builder)
    is_tidx_x = tid(0, _builder=_builder)
    is_bidx_x0 = is_bidx_x.__eq__(0, _builder=_builder)
    is_bidx_y0 = is_bidx_y.__eq__(0, _builder=_builder)
    is_bidx_z0 = is_bidx_z.__eq__(0, _builder=_builder)
    is_tidx_x0 = is_tidx_x.__eq__(0, _builder=_builder)

    is_thread0 = is_bidx_x0.__and__(is_bidx_y0, _builder=_builder)
    is_thread0 = is_thread0.__and__(is_bidx_z0, _builder=_builder)
    is_thread0 = is_thread0.__and__(is_tidx_x0, _builder=_builder)
    return is_thread0

@builtin
def warp_id(_builder=None):
    return semantic.warp_id(_builder)

@builtin
def warpgroup_id(_builder=None):
    return semantic.warpgroup_id(_builder)

@builtin
def print(prefix_or_data, data=None, _builder=None):
    import string
    prefix_or_data = _constexpr_to_value(prefix_or_data)
    if isinstance(prefix_or_data, str):
        prefix = prefix_or_data
    else:
        prefix = "[some value]: "
        data = prefix_or_data

    b_ascii = True
    for ch in prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{prefix} is not an ascii string"

    if data is None:
        data = 0
    return semantic.device_print(prefix, [semantic.to_tensor(data, _builder)], False, _builder)

@builtin
def load0(pointer, space='auto', swizzle=(1, 1, 1), row_major=False, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _builder=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param space: The space of loaded tensor, either 'auto' or 'smem'
    :type space: `str`, 'auto' or 'smem'
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # `mask` and `other` can be constexpr
    mask = _constexpr_to_value(mask)
    other = _constexpr_to_value(other)
    if mask is not None:
        mask = semantic.to_tensor(mask, _builder)
    if other is not None:
        other = semantic.to_tensor(other, _builder)
    padding_option = _constexpr_to_value(padding_option)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)

    space = _constexpr_to_value(space)
    swizzle = _constexpr_to_value(swizzle)
    row_major = _constexpr_to_value(row_major)

    new_ptr = semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                         volatile, _builder)
    if space != 'auto':
        return semantic.local_alloc(_builder, new_ptr, space=space, swizzle=swizzle, row_major=row_major)
    else:
        return new_ptr
