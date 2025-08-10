import triton.language as tl
from triton.language.core import builtin, _shape_check_impl, _unwrap_if_constexpr
from typing import Sequence

@builtin
def tid(axis, _semantic=None):
    axis = _unwrap_if_constexpr(axis)
    return _semantic.threadIdx(axis)

@builtin
def tdim(axis, _semantic=None):
    axis = _unwrap_if_constexpr(axis)
    return _semantic.blockDim(axis)

@builtin
def bid(axis, _semantic=None):
    axis = _unwrap_if_constexpr(axis)
    return _semantic.blockIdx(axis)

@builtin
def bdim(axis, _semantic=None):
    axis = _unwrap_if_constexpr(axis)
    return _semantic.gridDim(axis)

@builtin
def thread0(_semantic=None):
    is_bidx_x = bid(0, _semantic=_semantic)
    is_bidx_y = bid(1, _semantic=_semantic)
    is_bidx_z = bid(2, _semantic=_semantic)
    is_tidx_x = tid(0, _semantic=_semantic)
    is_bidx_x0 = is_bidx_x.__eq__(0, _semantic=_semantic)
    is_bidx_y0 = is_bidx_y.__eq__(0, _semantic=_semantic)
    is_bidx_z0 = is_bidx_z.__eq__(0, _semantic=_semantic)
    is_tidx_x0 = is_tidx_x.__eq__(0, _semantic=_semantic)

    is_thread0 = is_bidx_x0.__and__(is_bidx_y0, _semantic=_semantic)
    is_thread0 = is_thread0.__and__(is_bidx_z0, _semantic=_semantic)
    is_thread0 = is_thread0.__and__(is_tidx_x0, _semantic=_semantic)
    return is_thread0

@builtin
def wg_thread0(wgid, _semantic=None):
    is_bidx_x = bid(0, _semantic=_semantic)
    is_bidx_y = bid(1, _semantic=_semantic)
    is_bidx_z = bid(2, _semantic=_semantic)
    is_tidx_x = tid(0, _semantic=_semantic)

    thread_id = _unwrap_if_constexpr(wgid * 128)

    is_bidx_x0 = is_bidx_x.__eq__(0, _semantic=_semantic)
    is_bidx_y0 = is_bidx_y.__eq__(0, _semantic=_semantic)
    is_bidx_z0 = is_bidx_z.__eq__(0, _semantic=_semantic)
    is_tidx_x0 = is_tidx_x.__eq__(thread_id, _semantic=_semantic)

    is_thread0 = is_bidx_x0.__and__(is_bidx_y0, _semantic=_semantic)
    is_thread0 = is_thread0.__and__(is_bidx_z0, _semantic=_semantic)
    is_thread0 = is_thread0.__and__(is_tidx_x0, _semantic=_semantic)
    return is_thread0

@builtin
def warp_id(_semantic=None):
    return _semantic.warp_id()

@builtin
def warpgroup_id(_semantic=None):
    return _semantic.warpgroup_id()

@builtin
def is_warpgroup(ids, _semantic=None):
    return _semantic.is_warpgroup(ids)

@builtin
def reg_alloc(count, _semantic=None):
    count = _unwrap_if_constexpr(count)
    return _semantic.reg_alloc(count)

@builtin
def reg_dealloc(count, _semantic=None):
    count = _unwrap_if_constexpr(count)
    return _semantic.reg_dealloc(count)

@builtin
def smem_alloc(shape, dtype: tl.dtype, num_stages:int=1, mutable:bool=True, _semantic=None) -> tl.tensor:
    shape = _shape_check_impl(shape)
    dtype = _unwrap_if_constexpr(dtype)
    num_stages = _unwrap_if_constexpr(num_stages)
    mutable = _unwrap_if_constexpr(mutable)
    return _semantic.smem_alloc(shape, dtype, num_stages, mutable)

@builtin
def mbar_alloc(arr_count: int, num_stages:int=1, _semantic=None) -> tl.tensor:
    arr_count = _unwrap_if_constexpr(arr_count)
    num_stages = _unwrap_if_constexpr(num_stages)
    return _semantic.mbar_alloc(arr_count, num_stages)

@builtin
def tma_load(value: tl.tensor, desc, offsets, mbar:tl.tensor, _semantic=None) -> tl.tensor:
    return _semantic.tma_load(value, desc, offsets, mbar, "", "")

@builtin
def tma_gather(value: tl.tensor, desc, mbar: tl.tensor, *args, _semantic=None) -> tl.tensor:
    """Gather multiple descriptors worth of data"""
    assert len(args) == 2, f"descriptor gather only supports 2D indexing, but got {len(args)}"
    x_offsets = args[0]
    y_offset = args[1]
    return _semantic.tma_gather(value, desc, x_offsets, y_offset, mbar, "", "")

@builtin
def get_buffer(src: tl.tensor, index, _semantic=None) -> tl.tensor:
    return _semantic.get_buffer(src, index)

@builtin
def mbar_expect(mbar: tl.tensor, size_in_bytes: int, pred: tl.tensor=None, _semantic=None) -> tl.tensor:
    size_in_bytes = _unwrap_if_constexpr(size_in_bytes)
    if pred is None:
        pred = tl.full((), True, dtype=tl.int1, _semantic=_semantic)
    return _semantic.mbar_expect(mbar, size_in_bytes, pred)

@builtin
def mbar_wait(mbar: tl.tensor, phase, _semantic=None) -> tl.tensor:
    return _semantic.mbar_wait(mbar, phase)

@builtin
def mbar_arrive(mbar: tl.tensor, pred: tl.tensor=None,
                track_async_op:bool=False, tx_cnt:int =0,
                _semantic=None) -> tl.tensor:
    track_async_op = _unwrap_if_constexpr(track_async_op)
    tx_cnt = _unwrap_if_constexpr(tx_cnt)
    # pred is Value not const expr
    if pred is None:
        pred = tl.full((), True, dtype=tl.int1, _semantic=_semantic)
    return _semantic.mbar_arrive(mbar, pred, track_async_op, tx_cnt)

@builtin
def dot_wait(pendings: int, _semantic=None) -> tl.tensor:
    pendings = _unwrap_if_constexpr(pendings)
    return _semantic.dot_wait(pendings)

@builtin
def bar_arrive(bar: int, num_threads: int, _semantic=None) -> tl.tensor:
    bar = _unwrap_if_constexpr(bar)
    num_threads = _unwrap_if_constexpr(num_threads)
    return _semantic.bar_arrive(bar, num_threads)

@builtin
def bar_wait(bar: int, num_threads: int, _semantic=None) -> tl.tensor:
    bar = _unwrap_if_constexpr(bar)
    num_threads = _unwrap_if_constexpr(num_threads)
    return _semantic.bar_wait(bar, num_threads)

@builtin
def print(prefix_or_data, data=None, _semantic=None):
    import string
    prefix_or_data = _unwrap_if_constexpr(prefix_or_data)
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
    return _semantic.device_print(prefix, [_semantic.to_tensor(data)], False)

@builtin
def async_load(mem, pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _semantic=None):
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
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", "ca", "cg"}, where "ca" stands for
        cache at all levels and "cg" stands for cache at global level (cache in L2 and below, not L1), see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # `mask` and `other` can be constexpr
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)
    padding_option = _unwrap_if_constexpr(padding_option)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    return _semantic.async_load(mem, pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                         volatile)

@builtin
def async_load_wait(async_token: tl.tensor, num: int, _semantic=None) -> tl.tensor:
    num = _unwrap_if_constexpr(num)
    return _semantic.async_load_wait(async_token, num)
