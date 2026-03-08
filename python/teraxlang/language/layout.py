from enum import Enum
import math
import triton
import triton.language as tl
from triton.language.core import builtin, constexpr
from triton.language.core import _experimental_reinterpret_tensor_descriptor
from triton.language import semantic
from .utils import _constexpr_to_value, _apply_binary_method

def get_stride_from_shape_and_order(shape, order, _builder=None):
    assert len(shape) == len(order)
    stride = [constexpr(1)] * len(order)
    cnt = 0
    cur_st = constexpr(1)
    for i in range(len(order)):
        for p, (s, o) in enumerate(zip(shape, order)):
            if o == cnt:
                stride[p] = cur_st
                if _builder:
                    cur_st = _apply_binary_method('__mul__', cur_st, s, _builder=_builder)
                else:
                    cur_st *= s
                cnt += 1
                break
    assert cnt == len(order)
    stride = tuple(stride)
    if _builder:
        stride = tl.tuple(stride)
    return stride

class OrderType(Enum):
    RIGHT = 1
    LEFT = 2
    MATOP1 = 1
    MATOP2 = 3

def convert_order_type(ot, rank, _builder=None):
    order = list(range(rank))
    order.reverse()
    if ot == OrderType.RIGHT:
        pass
    elif ot == OrderType.MATOP1:
        pass
    elif ot == OrderType.LEFT:
        order.reverse()
    elif ot == OrderType.MATOP2:
        order[-2], order[-1] = (order[-1], order[-2])
    order = tuple(order)
    if _builder:
        order = tl.tuple([constexpr(i) for i in order])
    return order

def Layout(shape, block_shape=None, order=OrderType.RIGHT, order_map=OrderType.RIGHT):
    """
    order can be None, but not stride.
    if stride is None, use order to infer
    """
    assert not (isinstance(order_map, OrderType) and block_shape is None)
    block_rank = len(block_shape) if block_shape is not None else len(order_map)
    if isinstance(order, OrderType):
        order = convert_order_type(order, len(shape))
    strides = get_stride_from_shape_and_order(shape, order)
    if isinstance(order_map, OrderType):
        order_map = convert_order_type(order_map, block_rank)
    all_rank = len(shape)
    sub_rank = block_rank
    assert all_rank >= sub_rank
    shape0 = []
    strides0 = []
    order0 = []
    shape1, strides1, order1 = (shape, strides, order)
    if all_rank > sub_rank:
        rank0 = all_rank - sub_rank
        shape0 = shape[:rank0]
        strides0 = strides[:rank0]
        order0 = order[:rank0]
        shape1 = shape[rank0:]
        strides1 = strides[rank0:]
        order1 = order[rank0:]
    assert len(order_map) == len(order1)
    ordered_shape = []
    ordered_strides = []
    for bo in order_map:
        for sh, st, o in zip(shape1, strides1, order1):
            o = _constexpr_to_value(o)
            bo = _constexpr_to_value(bo)
            if o == bo:
                ordered_shape.append(sh)
                ordered_strides.append(st)
    ordered_shape = tuple(ordered_shape)
    ordered_strides = tuple(ordered_strides)
    return (shape0, strides0, order0, ordered_shape, ordered_strides, block_shape, order_map)
HAS_TMA_DESC = 'nv_tma_desc_type' in dir(tl)
if HAS_TMA_DESC:
    import triton.tools.experimental_descriptor
import torch

class TmaDescKernelParam:
    TMA_DESC_SIZE = 128

    def __init__(self):
        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.int8, device='cpu')

    def fill_1d(self, ptr, dims, block_dims, element_size):
        assert len(dims) == len(block_dims)
        assert len(dims) == 1
        assert self.desc.data_ptr() % 64 == 0
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dims[0], block_dims[0], element_size, self.desc.data_ptr())

    def fill_2d(self, ptr, dims, block_dims, element_size):
        assert len(dims) == len(block_dims)
        assert len(dims) == 2
        assert self.desc.data_ptr() % 64 == 0
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dims[0], dims[1], block_dims[0], block_dims[1], element_size, self.desc.data_ptr())

    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()

def TmaDesc(tensor, layout):
    desc = TmaDescKernelParam()
    shape0, strides0, order0, ordered_shape, ordered_strides, _, order_map = layout
    y_dim = math.prod(shape0) * ordered_shape[0]
    if len(ordered_shape) == 1:
        dims = [y_dim]
        hook = lambda block_shape: desc.fill_1d(tensor.data_ptr(), dims, block_shape, tensor.element_size())
    elif len(ordered_shape) == 2:
        dims = [y_dim, ordered_shape[1]]
        hook = lambda block_shape: desc.fill_2d(tensor.data_ptr(), dims, block_shape, tensor.element_size())
    else:
        raise ValueError('block shape must be 1 or 2 for tma')
    return (desc, hook)

def compress_offsets(offsets, layout, rank, _builder):
    shape0, strides0, order0, ordered_shape, ordered_strides, _, order_map = layout
    remain_offset = len(shape0) + len(ordered_shape) - len(offsets)
    assert len(shape0) + len(ordered_shape) - len(offsets) >= 0
    offsets = tl.tuple(list(offsets.values) + [constexpr(0)] * remain_offset)
    all_rank = len(shape0) + len(ordered_shape)
    assert all_rank == len(offsets)
    if all_rank > rank:
        rank0 = all_rank - rank
        offsets0 = offsets[:rank0]
        strides0 = (list(strides0) + list(ordered_strides))[:rank0]
        offsets = offsets[rank0:]
        co = 0
        for o, s in zip(offsets0, strides0):
            cur_o = _apply_binary_method('__mul__', o, s, _builder=_builder)
            co = _apply_binary_method('__add__', cur_o, co, _builder=_builder)
        co = _apply_binary_method('__floordiv__', co, strides0[-1], _builder=_builder)
        return (co, offsets)
    return (tl.constexpr(0), offsets)

@builtin
def subtma(ptr, offsets, layout, block_shape, dtype, _builder=None):
    co, offsets = compress_offsets(offsets, layout, len(block_shape) - 1, _builder=_builder)
    desc = _experimental_reinterpret_tensor_descriptor(ptr, block_shape, dtype, _builder=_builder)
    return (desc, tl.tuple([co] + list(offsets)))

@builtin
def subtensor(ptr, offsets, layout, block_shape, _builder=None):
    """
    Assume:
        1. block are for the last dims of parent tensor, not in the middle
        2. offsets not specified are appended as 0 at last
    """
    shape0, strides0, order0, ordered_shape, ordered_strides, _, order_map = layout
    new_order_map = []
    for i in order_map:
        if isinstance(i, tl.tensor):
            i = 0
        new_order_map.append(i)
    order_map = new_order_map
    remain_offset = len(shape0) + len(ordered_shape) - len(offsets)
    assert len(shape0) + len(ordered_shape) - len(offsets) >= 0
    offsets = tl.tuple(list(offsets.values) + [constexpr(0)] * remain_offset)
    all_rank = len(shape0) + len(ordered_shape)
    sub_rank = len(ordered_shape)
    assert all_rank == len(offsets)
    if all_rank > sub_rank:
        rank0 = all_rank - sub_rank
        offsets0 = offsets[:rank0]
        offsets = offsets[rank0:]
        for o, s in zip(offsets0, strides0):
            cur_o = _apply_binary_method('__mul__', o, s, _builder=_builder)
            ptr = _apply_binary_method('__add__', ptr, cur_o, _builder=_builder)
    block_ptr = semantic.make_block_ptr(base=ptr, shape=ordered_shape, strides=ordered_strides, offsets=offsets, block_shape=block_shape, order=order_map, builder=_builder)
    return block_ptr

@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False, _builder=None):
    if isinstance(pointer, tl.tuple) and isinstance(pointer[0], tl.core._experimental_tensor_descriptor_base):
        desc, offsets = pointer
        return desc.load(offsets, _builder=_builder)
    return tl.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy, volatile, _builder=_builder)

@tl.core._tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='', _builder=None):
    if isinstance(pointer, tl.tuple) and isinstance(pointer[0], tl.core._experimental_tensor_descriptor_base):
        desc, offsets = pointer
        return desc.store(offsets, value, _builder=_builder)
    return tl.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, _builder=_builder)

def Layout0(shape, order=OrderType.RIGHT, _builder=None):
    """
    order can be None, but not stride.
    if stride is None, use order to infer
    """
    stride = None
    if _builder:
        shape = tl.tuple(shape)
    if isinstance(order, OrderType):
        order = convert_order_type(order, len(shape), _builder)
    if stride is None:
        stride = get_stride_from_shape_and_order(shape, order, _builder=_builder)
    res = (shape, stride, order)
    if _builder:
        res = tl.tuple(res)
    return res

@builtin
def local_layout(shape, order=OrderType.RIGHT, _builder=None):
    """
    order can be None, but not stride.
    if stride is None, use order to infer
    """
    return Layout(shape, order, _builder)

@builtin
def subtensor0(ptr, offsets, layout, block_layout=None, block_shape=None, block_order=OrderType.RIGHT, _builder=None):
    """
    Assume:
        1. block are for the last dims of parent tensor, not in the middle
        2. offsets not specified are appended as 0 at last
    """
    assert block_layout is not None or (block_shape is not None and block_order is not None)
    if block_layout is not None:
        block_shape = block_layout[0]
        block_order = block_layout[2]
    if isinstance(block_order, OrderType):
        block_order = convert_order_type(block_order, len(block_shape))
    assert len(layout[0]) - len(offsets) >= 0
    offsets = tl.tuple(list(offsets.values) + [constexpr(0)] * (len(layout[0]) - len(offsets)))
    all_rank = len(layout[0])
    sub_rank = len(block_shape)
    assert all_rank == len(offsets)
    assert all_rank >= sub_rank
    shape, strides, order = layout
    if all_rank > sub_rank:
        rank0 = all_rank - sub_rank
        offsets0 = offsets[:rank0]
        strides0 = strides[:rank0]
        shape = shape[rank0:]
        strides = strides[rank0:]
        order = order[rank0:]
        offsets = offsets[rank0:]
        for o, s in zip(offsets0, strides0):
            cur_o = _apply_binary_method('__mul__', o, s, _builder=_builder)
            ptr = _apply_binary_method('__add__', ptr, cur_o, _builder=_builder)
    assert len(block_order) == len(shape)
    ordered_shape = []
    ordered_strides = []
    for bo in block_order:
        for sh, st, o in zip(shape, strides, order):
            o = _constexpr_to_value(o)
            bo = _constexpr_to_value(bo)
            if o == bo:
                ordered_shape.append(sh)
                ordered_strides.append(st)
    ordered_shape = tl.tuple(ordered_shape)
    ordered_strides = tl.tuple(ordered_strides)
    block_ptr = semantic.make_block_ptr(base=ptr, shape=ordered_shape, strides=ordered_strides, offsets=offsets, block_shape=block_shape, order=block_order, builder=_builder)
    return block_ptr