import triton.language as tl
from triton._C.libtriton import ir
from triton.language import core
from triton.language.core import (
    builtin,
    _shape_check_impl,
    _unwrap_if_constexpr,
    expand_dims,
    broadcast_to,
    _wrap_axis,
    _insertion_guard,
    dtype,
    block_type,
)
from triton.language.standard import (
    _elementwise_max,
    _sum_combine,
    _pick_sum_dtype,
    _argmax_combine_tie_break_fast,
    _argmax_combine_tie_break_left,
)
from triton import knobs
from typing import Sequence, List
from ..runtime.jit import jit
from ._layouts import DistributedLayout


class distributed_type(block_type):
    def __init__(self, element_ty: dtype, shape: List[int], layout):
        super().__init__(element_ty, shape)
        self.layout = layout
        self.name = f"<{self.shape}, {self.element_ty}, {self.layout}>"
        assert isinstance(layout, DistributedLayout)

    def to_ir(self, builder: ir.builder) -> ir.type:
        elem_ty = self.element_ty.to_ir(builder)
        layout = self.layout._to_ir(builder)
        return builder.get_distributed_ty(elem_ty, self.shape, layout)

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = "_".join(map(str, self.shape))
        layout = self.layout.mangle()
        return f"{elt}S{shape}SL{layout}L"

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return distributed_type(scalar_ty, self.shape, self.layout)


@builtin
def tid(axis, _semantic=None):
    """Get the thread index within a block along the specified axis.

    Args:
        axis: The axis (0, 1, or 2) to get the thread index for.

    Returns:
        The thread index along the given axis.
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.threadIdx(axis)


@builtin
def tdim(axis, _semantic=None):
    """Get the block dimension (number of threads) along the specified axis.

    Args:
        axis: The axis (0, 1, or 2) to get the block dimension for.

    Returns:
        The number of threads along the given axis.
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.blockDim(axis)


@builtin
def bid(axis, _semantic=None):
    """Get the block index within the grid along the specified axis.

    Args:
        axis: The axis (0, 1, or 2) to get the block index for.

    Returns:
        The block index along the given axis.
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.blockIdx(axis)


@builtin
def bdim(axis, _semantic=None):
    """Get the grid dimension (number of blocks) along the specified axis.

    Args:
        axis: The axis (0, 1, or 2) to get the grid dimension for.

    Returns:
        The number of blocks along the given axis.
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.gridDim(axis)


@builtin
def thread0(_semantic=None):
    """Check if the current thread is thread 0 in the entire grid.

    Returns:
        A boolean tensor that is True only for thread (block 0,0,0 x thread 0,0,0).
    """
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
    """Check if the current thread is the first thread of the specified warpgroup.

    Args:
        wgid: The warpgroup ID to check against.

    Returns:
        A boolean tensor that is True only for the first thread of the given warpgroup.
    """
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
    """Get the warp ID within a thread block.

    Returns:
        The warp ID (0-31 on typical GPUs).
    """
    return _semantic.warp_id()


@builtin
def warpgroup_id(_semantic=None):
    """Get the warpgroup ID within a thread block.

    Returns:
        The warpgroup ID (0-3 for typical warpgroup sizes).
    """
    return _semantic.warpgroup_id()


@builtin
def lane_id(_semantic=None):
    """Get the lane ID within a warp.

    Returns:
        The lane ID (0-31 on typical GPUs).
    """
    return _semantic.lane_id()


@builtin
def cta_rank(_semantic=None):
    """Get the CTA (Cooperative Thread Array) rank within the grid.

    Returns:
        The linear CTA rank.
    """
    return _semantic.cta_rank()


@builtin
def is_warpgroup(ids, _semantic=None):
    """Check if the current thread belongs to a specific warpgroup.
    Should always use this condition instead of a pure if wgid == 0

    Args:
        ids: The warpgroup ID or IDs to check.

    Returns:
        A boolean tensor indicating if the current thread is in the specified warpgroup(s).
    """
    return _semantic.is_warpgroup(ids)


@builtin
def is_warp(ids, _semantic=None):
    """Check if the current thread belongs to a specific warp.
    Should always use this condition instead of a pure if wid == 0

    Args:
        ids: The warp ID or IDs to check.

    Returns:
        A boolean tensor indicating if the current thread is in the specified warp(s).
    """
    return _semantic.is_warp(ids)


@builtin
def reg_alloc(count, _semantic=None):
    """Allocate a number of registers.

    Args:
        count: The number of registers to allocate.

    Returns:
        A handle to the allocated registers.
    """
    count = _unwrap_if_constexpr(count)
    return _semantic.reg_alloc(count)


@builtin
def reg_dealloc(count, _semantic=None):
    """Deallocate a number of registers.

    Args:
        count: The number of registers to deallocate.

    Returns:
        A handle to the deallocated registers.
    """
    count = _unwrap_if_constexpr(count)
    return _semantic.reg_dealloc(count)


@builtin
def smem_alloc(
    shape, dtype: tl.dtype, num_stages: int = 1, mutable: bool = True, shared_enc=None, _semantic=None
) -> tl.tensor:
    """Allocate shared memory.

    Args:
        shape: The shape of the shared memory tensor.
        dtype: The data type of the shared memory.
        num_stages: Number of pipeline stages (for double buffering).
        mutable: Whether the memory can be modified.
        shared_enc: Optional shared memory encoding.

    Returns:
        A tensor pointing to the allocated shared memory.
    """
    # shape = _shape_check_impl(shape)
    dtype = _unwrap_if_constexpr(dtype)
    num_stages = _unwrap_if_constexpr(num_stages)
    mutable = _unwrap_if_constexpr(mutable)
    if shared_enc is not None:
        shared_enc = _unwrap_if_constexpr(shared_enc)
    return _semantic.smem_alloc(shape, dtype, num_stages, mutable, shared_enc)


@builtin
def smem_load(smem, layout=None, cta_id=-1, _semantic=None) -> tl.tensor:
    """Load data from shared memory.

    Args:
        smem: The shared memory tensor to load from.
        layout: The layout for the loaded tensor.
        cta_id: The CTA ID for the load operation, for DSMEM copy.

    Returns:
        A tensor with the loaded data.
    """
    layout = _unwrap_if_constexpr(layout)
    cta_id = _unwrap_if_constexpr(cta_id)
    return _semantic.smem_load(smem, layout, cta_id)


@builtin
def smem_store(smem, value, cta_id=-1, _semantic=None) -> None:
    """Store data to shared memory.

    Args:
        smem: The shared memory tensor to store to.
        value: The value to store.
        cta_id: The CTA ID for the store operation, for DSMEM copy.
    """
    cta_id = _unwrap_if_constexpr(cta_id)
    return _semantic.smem_store(smem, value, cta_id)


@builtin
def tmem_alloc(
    shape, dtype: tl.dtype, num_stages: int = 1, mutable: bool = True, shared_enc=None, _semantic=None
) -> tl.tensor:
    """Allocate tensor memory (TMEM).

    Args:
        shape: The shape of the tensor memory.
        dtype: The data type of the tensor memory.
        num_stages: Number of pipeline stages (for double buffering).
        mutable: Whether the memory can be modified.
        shared_enc: Optional shared memory encoding.

    Returns:
        A tensor pointing to the allocated tensor memory.
    """
    # shape = _shape_check_impl(shape)
    dtype = _unwrap_if_constexpr(dtype)
    num_stages = _unwrap_if_constexpr(num_stages)
    mutable = _unwrap_if_constexpr(mutable)
    return _semantic.tmem_alloc(shape, dtype, num_stages, mutable, shared_enc)


@builtin
def tmem_load(smem, cta_id=-1, _semantic=None) -> tl.tensor:
    """Load data from tensor memory (TMEM).

    Args:
        smem: The tensor memory to load from.
        cta_id: The CTA ID for the load operation.

    Returns:
        A tensor with the loaded data.
    """
    cta_id = _unwrap_if_constexpr(cta_id)
    return _semantic.tmem_load(smem, cta_id)


@builtin
def tmem_store(smem, value, cta_id=-1, _semantic=None) -> None:
    """Store data to tensor memory (TMEM).

    Args:
        smem: The tensor memory to store to.
        value: The value to store.
        cta_id: The CTA ID for the store operation.
    """
    cta_id = _unwrap_if_constexpr(cta_id)
    return _semantic.tmem_store(smem, value, cta_id)


@builtin
def frag_smem_load(
    smem, shape, layout, other=None, pred=None, is_broadcast=False, cta_id=-1, _semantic=None
) -> tl.tensor:
    """DEPRECATED: Load a fragment of shared memory with support for lane/warp-level predication and broadcasting.

    Args:
        smem: The shared memory tensor to load from.
        shape: The shape of the fragment.
        layout: The layout for the loaded tensor.
        other: Optional fill value for non-participating threads.
        pred: Optional predicate for lane/warp-level selection.
        is_broadcast: Deprecated, use smem_load/smem_store instead.
        cta_id: The CTA ID for the load operation.

    Returns:
        A tensor with the loaded fragment data.
    """
    layout = _unwrap_if_constexpr(layout)
    shape = _shape_check_impl((shape))
    if other is not None:
        other = _semantic.to_tensor(other)
    is_broadcast = _unwrap_if_constexpr(is_broadcast)
    if is_broadcast:
        print("DEPRECATED: is_broadcast should be well covered by smem_load/smem_store, should not use frag version")
    assert not (other is not None and is_broadcast), "fill (other) and broadcast can not be specified at the same time"
    cta_id = _unwrap_if_constexpr(cta_id)
    return _semantic.frag_smem_load(smem, shape, layout, other, pred, is_broadcast, cta_id)


@builtin
def frag_smem_store(smem, value, layout, pred=None, cta_id=-1, mbar=None, predStr: str = "", _semantic=None) -> None:
    """DEPRECATED: Store a fragment to shared memory with 2D -> squeezed 1D register-based storage.

    Args:
        smem: The shared memory tensor to store to.
        value: The value to store.
        layout: The layout for the stored tensor.
        pred: Optional predicate for lane/warp-level selection.
        cta_id: The CTA ID for the store operation.
        mbar: Optional memory barrier.
        predStr: Optional predicate string (e.g., "slice:1") to derive subRegLayout automatically.

    Note:
        For other cases, please use smem_store.
        It allows fractional store from the whole tensor.
    """
    if not isinstance(value.type, block_type):
        value = core.full((1,), value, value.type, _semantic=_semantic)
    layout = _unwrap_if_constexpr(layout)
    cta_id = _unwrap_if_constexpr(cta_id)
    pred = _unwrap_if_constexpr(pred)
    predStr = _unwrap_if_constexpr(predStr)
    return _semantic.frag_smem_store(smem, value, layout, pred, cta_id, mbar, predStr)


@builtin
def fence_proxy_async(_semantic=None):
    """Insert an asynchronous fence proxy operation."""
    _semantic.fence_proxy_async()


@builtin
def relayout(value, shape, layout, _semantic=None) -> tl.tensor:
    """DEPRECATED: Relayout a tensor to a new shape and layout.

    Args:
        value: The tensor to relayout.
        shape: The new shape.
        layout: The new layout.

    Note:
        Must be called after all reshape operations.
        Must be called after all cast operations.
    """
    layout = _unwrap_if_constexpr(layout)
    shape = _shape_check_impl(shape)
    return _semantic.relayout(value, shape, layout)


@builtin
def print_layout(shape, dtype, layout, save_loc=None, _semantic=None):
    """Print the linear layout for debugging purposes.

    Args:
        shape: The shape of the tensor.
        dtype: The data type.
        layout: The layout to convert.
        save_loc: Optional location to save the layout.
    """
    shape = _shape_check_impl(shape)
    layout = _unwrap_if_constexpr(layout)
    dtype = _unwrap_if_constexpr(dtype)
    return _semantic.to_linear_layout(shape, dtype, layout, save_loc)


@builtin
def mbar_alloc(arr_count: int, num_stages: int = 1, _semantic=None) -> tl.tensor:
    """Allocate a mbar (mbarrier) for thread synchronization.

    Args:
        arr_count: Number of array elements.
        num_stages: Number of pipeline stages.

    Returns:
        A tensor representing the allocated mbar.
    """
    arr_count = _unwrap_if_constexpr(arr_count)
    num_stages = _unwrap_if_constexpr(num_stages)
    return _semantic.mbar_alloc(arr_count, num_stages)


@builtin
def tma_load(value: tl.tensor, desc, offsets, mbar: tl.tensor, contiguity: int = -1, _semantic=None) -> tl.tensor:
    """Load data using Tensor Memory Accelerator (TMA).

    Args:
        value: The tensor to store the loaded data.
        desc: The TMA descriptor.
        offsets: The offsets for the load.
        mbar: The mbarrier for synchronization.
        contiguity: The contiguity parameter.

    Returns:
        A tensor with the loaded data.
    """
    contiguity = _unwrap_if_constexpr(contiguity)
    return _semantic.tma_load(value, desc, offsets, mbar, "", "", contiguity)


@builtin
def tma_gather(value: tl.tensor, desc, mbar: tl.tensor, *args, _semantic=None) -> tl.tensor:
    """Gather multiple descriptors worth of data using TMA.

    Args:
        value: The tensor to store the gathered data.
        desc: The TMA descriptor.
        mbar: The mbarrier for synchronization.
        *args: Must be (x_offsets, y_offset) for 2D indexing.

    Returns:
        A tensor with the gathered data.
    """
    assert len(args) == 2, f"descriptor gather only supports 2D indexing, but got {len(args)}"
    x_offsets = args[0]
    y_offset = args[1]
    return _semantic.tma_gather(value, desc, x_offsets, y_offset, mbar, "", "")


@builtin
def tma_store(value: tl.tensor, desc, offsets, _semantic=None) -> tl.tensor:
    """Store data using Tensor Memory Accelerator (TMA).

    Args:
        value: The tensor to store.
        desc: The TMA descriptor.
        offsets: The offsets for the store.

    Returns:
        A tensor representing the store operation.
    """
    return _semantic.tma_store(value, desc, offsets)


@builtin
def tma_store_wait(pendings: int, _semantic=None) -> tl.tensor:
    """Wait for pending TMA store operations.

    Args:
        pendings: The number of pending operations to wait for.

    Returns:
        A tensor representing the wait operation.
    """
    return _semantic.tma_store_wait(pendings)


@builtin
def get_buffer(src: tl.tensor, index, _semantic=None) -> tl.tensor:
    """Get a buffer from a tensor at the specified index.

    Args:
        src: The source tensor.
        index: The buffer index.

    Returns:
        The buffer tensor at the specified index.
    """
    return _semantic.get_buffer(src, index)


@builtin
def mbar_expect(mbar: tl.tensor, size_in_bytes: int, pred: tl.tensor = None, _semantic=None) -> tl.tensor:
    """Expect data in a mbar (mbarrier).

    Args:
        mbar: The mbarrier tensor.
        size_in_bytes: The expected size in bytes.
        pred: Optional predicate for the operation.

    Returns:
        A tensor representing the expect operation.
    """
    size_in_bytes = _unwrap_if_constexpr(size_in_bytes)
    if pred is None:
        pred = tl.full((), True, dtype=tl.int1, _semantic=_semantic)
    return _semantic.mbar_expect(mbar, size_in_bytes, pred)


@builtin
def mbar_wait(mbar: tl.tensor, phase, _semantic=None) -> tl.tensor:
    """Wait on a mbar (mbarrier) for a specific phase.

    Args:
        mbar: The mbarrier tensor.
        phase: The phase to wait for.

    Returns:
        A tensor representing the wait operation.
    """
    return _semantic.mbar_wait(mbar, phase)


@builtin
def mbar_arrive(
    mbar: tl.tensor, pred: tl.tensor = None, track_async_op: bool = False, tx_cnt: int = 0, _semantic=None
) -> tl.tensor:
    """Arrive at a mbar (mbarrier) for synchronization.

    Args:
        mbar: The mbarrier tensor.
        pred: Optional predicate for the operation.
        track_async_op: Whether to track async operations.
        tx_cnt: Transaction count.

    Returns:
        A tensor representing the arrive operation.
    """
    track_async_op = _unwrap_if_constexpr(track_async_op)
    tx_cnt = _unwrap_if_constexpr(tx_cnt)
    # pred is Value not const expr
    if pred is None:
        pred = tl.full((), True, dtype=tl.int1, _semantic=_semantic)
    return _semantic.mbar_arrive(mbar, pred, track_async_op, tx_cnt)


@builtin
def dot_wait(pendings: int, _semantic=None) -> tl.tensor:
    """Wait for pending dot product operations.

    Args:
        pendings: The number of pending operations to wait for.

    Returns:
        A tensor representing the wait operation.
    """
    pendings = _unwrap_if_constexpr(pendings)
    return _semantic.dot_wait(pendings)


@builtin
def bar_arrive(bar: int, num_threads: int, _semantic=None) -> tl.tensor:
    """Arrive at a barrier.

    Args:
        bar: The barrier ID.
        num_threads: The number of threads participating in the barrier.

    Returns:
        A tensor representing the arrive operation.
    """
    bar = _unwrap_if_constexpr(bar)
    num_threads = _unwrap_if_constexpr(num_threads)
    return _semantic.bar_arrive(bar, num_threads)


@builtin
def bar_wait(bar: int, num_threads: int, _semantic=None) -> tl.tensor:
    """Wait at a barrier.

    Args:
        bar: The barrier ID.
        num_threads: The number of threads participating in the barrier.

    Returns:
        A tensor representing the wait operation.
    """
    bar = _unwrap_if_constexpr(bar)
    num_threads = _unwrap_if_constexpr(num_threads)
    return _semantic.bar_wait(bar, num_threads)


@builtin
def print(prefix_or_data, data=None, _semantic=None):
    """Print data from the GPU kernel.

    Args:
        prefix_or_data: Either a string prefix or the data to print.
        data: Optional data to print (if prefix_or_data is a string).

    Note:
        The prefix must be an ASCII string.
    """
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
def async_load(
    mem,
    pointer,
    mask=None,
    other=None,
    boundary_check=(),
    padding_option="",
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    contiguity=-1,
    _semantic=None,
):
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
    contiguity = _unwrap_if_constexpr(contiguity)
    return _semantic.async_load(
        mem, pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy, volatile, contiguity
    )


@builtin
def async_load_wait(pendings: int, _semantic=None) -> tl.tensor:
    """Wait for pending async load operations.

    Args:
        pendings: The number of pending operations to wait for.

    Returns:
        A tensor representing the wait operation.
    """
    pendings = _unwrap_if_constexpr(pendings)
    return _semantic.async_load_wait(pendings)


@builtin
def smem_index(input, index: int, _semantic=None) -> tl.tensor:
    """Get an indexed view of shared memory.

    Args:
        input: The shared memory tensor.
        index: The index to access.

    Returns:
        A tensor representing the indexed view.
    """
    index = _unwrap_if_constexpr(index)
    return _semantic.smem_index(input, index)


@builtin
def smem_slice(input, start, length, dim, _semantic=None) -> tl.tensor:
    """Get a slice of shared memory.

    Args:
        input: The shared memory tensor.
        start: The starting index.
        length: The length of the slice.
        dim: The dimension to slice.

    Returns:
        A tensor representing the sliced view.
    """
    start = _unwrap_if_constexpr(start)
    length = _unwrap_if_constexpr(length)
    dim = _unwrap_if_constexpr(dim)
    return _semantic.smem_slice(input, start, length, dim)


@builtin
def smem_trans(input, order, _semantic=None) -> tl.tensor:
    """Transpose shared memory.

    Args:
        input: The shared memory tensor.
        order: The transposition order.

    Returns:
        A tensor representing the transposed view.
    """
    order = _shape_check_impl(order)
    return _semantic.smem_trans(input, order)


@builtin
def smem_reshape(self, input, shape, _semantic=None) -> tl.tensor:
    """Reshape shared memory.

    Args:
        input: The shared memory tensor.
        shape: The new shape.

    Returns:
        A tensor representing the reshaped view.
    """
    shape = _shape_check_impl(shape)
    return _semantic.smem_reshape(input, shape)


from triton.language import tensor


@builtin
def warp_reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int | None
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool

    """
    if isinstance(input, tensor):
        return warp_reduce((input,), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic, _generator=_generator)[
            0
        ]

    def make_combine_region(reduce_op):
        param_types = [t.type.scalar for t in input] * 2
        region = reduce_op.get_region(0)
        builder = _semantic.builder
        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _semantic=_semantic)
        return t

    axis = _unwrap_if_constexpr(axis)
    keep_dims = _unwrap_if_constexpr(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = _semantic.warp_reduction(input, axis, make_combine_region)
    if keep_dims:
        if axis is not None:
            ret = tuple(expand_dims(t, axis, _semantic=_semantic) for t in ret)
        else:
            ret = tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def _warp_reduce_with_indices(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    """Internal warp reduction that also returns indices.

    Args:
        input: The input tensor.
        axis: The dimension along which to reduce.
        combine_fn: The combination function.
        keep_dims: Whether to keep reduced dimensions.

    Returns:
        A tuple of (reduced values, indices).
    """
    axis = _unwrap_if_constexpr(axis)
    n = input.shape[axis]
    index = arange(0, n, _semantic=_semantic)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [constexpr(d) for d in builtins.range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _semantic=_semantic)
        index = broadcast_to(index, input.shape, _semantic=_semantic)

    rvalue, rindices = warp_reduce(
        (input, index), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic, _generator=_generator
    )
    return rvalue, rindices


@builtin
def dotx(
    input,
    other,
    acc=None,
    useD=None,
    pred=None,
    mbars=[],
    mbarPreds=[],
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=core.float32,
    _semantic=None,
):
    """
    Returns the matrix product of two blocks. Especially useful for tcgen05.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
    For three-dimensional blocks, `tl.dot` performs the batched matrix product,
    where the first dimension of each block represents the batch dimension.

    :param input: The first tensor to be multiplied.
    :type input: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :type acc: 2D or 3D tensor of scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}
    :param input_precision: How to exercise the Tensor Cores for f32 x f32. If
      the device does not have Tensor Cores or the inputs are not of dtype f32,
      this option is ignored. For devices that do have tensor cores, the
      default precision is tf32.
    :type input_precision: string. Available options for nvidia: :code:`"tf32"`, :code:`"tf32x3"`, :code:`"ieee"`. Default: :code:`"tf32"`. Available options for amd: :code:`"ieee"`, (CDNA3 only) :code:`"tf32"`.
    :param allow_tf32: *Deprecated.* If true, input_precision is set to "tf32".
      Only one of :code:`input_precision` and :code:`allow_tf32` can be
      specified (i.e. at least one must be :code:`None`).
    """
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or (
            "tf32" if (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
        )

    input_precision = _unwrap_if_constexpr(input_precision)
    out_dtype = _unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = _unwrap_if_constexpr(max_num_imprecise_acc)
    acc = _unwrap_if_constexpr(acc)
    mbars = [_unwrap_if_constexpr(mbar) for mbar in mbars]
    mbarPreds = [_unwrap_if_constexpr(mbarPred) for mbarPred in mbarPreds]
    useD = _unwrap_if_constexpr(useD)
    pred = _unwrap_if_constexpr(pred)
    return _semantic.dotx(
        input, other, acc, useD, pred, mbars, mbarPreds, input_precision, max_num_imprecise_acc, out_dtype
    )


### from standard


@jit
def warp_max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    """Compute the maximum value across a warp.

    Args:
        input: The input tensor.
        axis: The dimension along which to reduce. If None, reduces across all dimensions.
        return_indices: Whether to return the indices of maximum values.
        return_indices_tie_break_left: If True, break ties by choosing leftmost index.
        keep_dims: Whether to keep reduced dimensions with size 1.

    Returns:
        The maximum value, or a tuple of (value, indices) if return_indices is True.
    """
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return _warp_reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return _warp_reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return warp_reduce(input, axis, _elementwise_max, keep_dims=keep_dims)


@jit
def warp_sum(input, axis=None, keep_dims=False, dtype: core.constexpr = None):
    """Compute the sum across a warp.

    Args:
        input: The input tensor.
        axis: The dimension along which to reduce. If None, reduces across all dimensions.
        keep_dims: Whether to keep reduced dimensions with size 1.
        dtype: Optional output dtype for the reduction.

    Returns:
        The sum of all elements along the specified axis.
    """
    # Pick a default dtype for the reduction if one was not specified.
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)
    return warp_reduce(input, axis, _sum_combine, keep_dims=keep_dims)
