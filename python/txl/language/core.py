import triton.language as tl
from triton._C.libtriton import ir
from triton.language import core
from triton.language.core import builtin, _shape_check_impl, _unwrap_if_constexpr, expand_dims, broadcast_to, _wrap_axis, _insertion_guard, \
        dtype, block_type
from triton.language.standard import _elementwise_max, _sum_combine, _pick_sum_dtype, _argmax_combine_tie_break_fast, _argmax_combine_tie_break_left
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
def lane_id(_semantic=None):
    return _semantic.lane_id()

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
def smem_load(smem, layout, _semantic=None) -> tl.tensor:
    layout = _unwrap_if_constexpr(layout)
    return _semantic.smem_load(smem, layout)

@builtin
def smem_store(smem, value, _semantic=None) -> None:
    return _semantic.smem_store(smem, value)

@builtin
def frag_smem_load(smem, layout, broadcast=False, _semantic=None) -> tl.tensor:
    layout = _unwrap_if_constexpr(layout)
    broadcast = _unwrap_if_constexpr(broadcast)
    return _semantic.frag_smem_load(smem, layout, broadcast)

@builtin
def frag_smem_store(smem, value, layout, _semantic=None) -> None:
    layout = _unwrap_if_constexpr(layout)
    return _semantic.frag_smem_store(smem, value, layout)

@builtin
def sub_layout(smem, value, layout, _semantic=None) -> tl.tensor:
    layout = _unwrap_if_constexpr(layout)
    return _semantic.sub_layout(smem, value, layout)

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
def async_load_wait(pendings: int, _semantic=None) -> tl.tensor:
    pendings = _unwrap_if_constexpr(pendings)
    return _semantic.async_load_wait(pendings)


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
        return warp_reduce((input, ), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic, _generator=_generator)[0]

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
    axis = _unwrap_if_constexpr(axis)
    n = input.shape[axis]
    index = arange(0, n, _semantic=_semantic)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [constexpr(d) for d in builtins.range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _semantic=_semantic)
        index = broadcast_to(index, input.shape, _semantic=_semantic)

    rvalue, rindices = warp_reduce((input, index), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic,
                              _generator=_generator)
    return rvalue, rindices


### from standard

@jit
def warp_max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
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
    # Pick a default dtype for the reduction if one was not specified.
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)
    return warp_reduce(input, axis, _sum_combine, keep_dims=keep_dims)
