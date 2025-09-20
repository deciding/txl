from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.semantic import TritonSemantic, TensorTy
from typing import Optional, Tuple, Sequence
from .core import distributed_type

class TXLSemantic(TritonSemantic):

    def _str_to_load_cache_modifierx(self, cache_modifier):
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

    def _str_to_eviction_policyx(self, eviction_policy):
        eviction = ir.EVICTION_POLICYX.NORMAL  # default
        if eviction_policy:
            if eviction_policy == "evict_last":
                eviction = ir.EVICTION_POLICYX.EVICT_LAST
            elif eviction_policy == "evict_first":
                eviction = ir.EVICTION_POLICYX.EVICT_FIRST
            else:
                raise ValueError(f"Eviction policy {eviction_policy} not supported")
        return eviction

    def _str_to_padding_optionx(self, padding_option):
        padding = None  # default
        if padding_option:
            if padding_option == "zero":
                padding = ir.PADDING_OPTIONX.PAD_ZERO
            elif padding_option == "nan":
                padding = ir.PADDING_OPTIONX.PAD_NAN
            else:
                raise ValueError(f"Padding option {padding_option} not supported")
        return padding


    def threadIdx(self, axis:int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"thread index axis must be 0, 1, or 2 but got {axis}")
        handle = self.builder.create_get_threadidx(axis)
        return self.tensor(self.builder.create_index_cast(handle, tl.int32.to_ir(self.builder)), tl.int32)

    def blockDim(self, axis:int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"block dim axis must be 0, 1, or 2 but got {axis}")
        handle = self.builder.create_get_blockdim(axis)
        return self.tensor(self.builder.create_index_cast(handle, tl.int32.to_ir(self.builder)), tl.int32)

    def blockIdx(self, axis:int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"block index axis must be 0, 1, or 2 but got {axis}")
        return self.tensor(self.builder.create_get_program_id(axis), tl.int32)

    def gridDim(self, axis:int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"grid dim axis must be 0, 1, or 2 but got {axis}")
        return self.tensor(self.builder.create_get_num_programs(axis), tl.int32)

    def warp_id(self) -> TensorTy:
        return self.tensor(self.builder.create_get_canonical_wrap_id(), tl.int32)

    def warpgroup_id(self) -> TensorTy:
        return self.tensor(self.builder.create_get_canonical_wrapgroup_id(), tl.int32)

    def lane_id(self) -> TensorTy:
        return self.tensor(self.builder.create_get_lane_id(), tl.int32)

    def is_warpgroup(self, ids) -> TensorTy:
        return self.tensor(self.builder.create_is_warpgroup(ids), tl.int1)

    def reg_alloc(self, count:int):
        self.builder.create_reg_alloc(count)

    def reg_dealloc(self, count:int):
        self.builder.create_reg_dealloc(count)

    def smem_alloc(self, shape, dtype: tl.dtype, num_stages:int=1, mutable:bool=True) -> TensorTy:
        block_type = tl.block_type(dtype, shape)
        dtype = dtype.to_ir(self.builder)
        return self.tensor(
            self.builder.create_smem_alloc(shape, dtype, num_stages, mutable), block_type)

    def smem_load(self, mem_desc, layout):
        ret_ty = tl.block_type(mem_desc.dtype, mem_desc.shape)
        reg_ty = distributed_type(mem_desc.dtype, mem_desc.shape, layout)
        handle = self.builder.create_smem_load(ret_ty.to_ir(self.builder), mem_desc.handle, reg_ty.to_ir(self.builder))
        return self.tensor(handle, ret_ty)

    def smem_store(self, mem_desc, value):
        assert value.shape == mem_desc.shape, f"source shape {value.shape} and destination shape {mem_desc.shape} must match"
        assert value.dtype == mem_desc.dtype, f"source dtype {value.dtype} and destination dtype {mem_desc.dtype} must match"
        self.builder.create_smem_store(mem_desc.handle, value.handle)

    def frag_smem_load(self, mem_desc, layout, other, layout_full, broadcast):
        if other is not None:
            shape = layout_full.shape()
        else:
            shape = mem_desc.shape
        ret_ty = tl.block_type(mem_desc.dtype, shape)
        reg_ty = distributed_type(mem_desc.dtype, mem_desc.shape, layout) # loading reg should keep the partial shape and layout
        handle = self.builder.create_frag_smem_load(ret_ty.to_ir(self.builder), mem_desc.handle, other.handle if other else None, reg_ty.to_ir(self.builder), broadcast)
        return self.tensor(handle, ret_ty)

    def frag_smem_store(self, mem_desc, value, layout):
        reg_ty = distributed_type(mem_desc.dtype, mem_desc.shape, layout)
        #assert value.shape == mem_desc.shape, f"source shape {value.shape} and destination shape {mem_desc.shape} must match"
        assert value.dtype == mem_desc.dtype, f"source dtype {value.dtype} and destination dtype {mem_desc.dtype} must match"
        self.builder.create_frag_smem_store(mem_desc.handle, value.handle, reg_ty.to_ir(self.builder))

    def sub_layout(self, mem_desc, value, layout):
        ret_ty = tl.block_type(mem_desc.dtype, mem_desc.shape)
        reg_ty = distributed_type(mem_desc.dtype, mem_desc.shape, layout)
        handle = self.builder.create_sub_layout(reg_ty.to_ir(self.builder), value.handle)
        return self.tensor(handle, ret_ty)

    def mbar_alloc(self, arr_count: int, num_stages:int=1) -> TensorTy:
        block_type = tl.block_type(tl.int64, [1])
        return self.tensor(
            self.builder.create_mbar_alloc(arr_count, num_stages), block_type)

    def tma_load(self, value: tl.tensor, desc: tl.tensor_descriptor_base, offsets,
            mbar: tl.tensor,
            cache_modifier: str, eviction_policy: str) -> TensorTy:
        assert isinstance(desc, tl.tensor_descriptor_base)
        ndim = len(desc.block_shape)
        assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        x = self.builder.create_tma_load(value.handle, mbar.handle, desc.handle, offsets, self._str_to_load_cache_modifierx(cache_modifier),
                                                self._str_to_eviction_policyx(eviction_policy))
        return self.tensor(x, tl.void)

    def tma_gather(self, value, desc, x_offsets, y_offset, mbar: tl.tensor, cache_modifier: str, eviction_policy: str) -> TensorTy:
        assert isinstance(desc, tl.tensor_descriptor_base)
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
        # TODO: check that value type is same as type
        #type.to_ir(self.builder)
        y_offset = self._convert_to_ir_values((y_offset, ), require_i64=False)[0]
        x = self.builder.create_tma_gather(value.handle, mbar.handle, desc.handle, x_offsets.handle, y_offset)
        return self.tensor(x, tl.void)

    def dot_wait(self, pendings:int) -> TensorTy:
        x = self.builder.create_dot_wait(pendings)
        return self.tensor(x, tl.void)

    def bar_arrive(self, bar: int, num_threads: int) -> TensorTy:
        x = self.builder.create_bar_arrive(bar, num_threads)
        return self.tensor(x, tl.void)

    def bar_wait(self, bar: int, num_threads: int) -> TensorTy:
        x = self.builder.create_bar_wait(bar, num_threads)
        return self.tensor(x, tl.void)

    def get_buffer(self, src: tl.tensor, index: tl.tensor) -> TensorTy:
        # TODO: check
        index = tl.tensor(self._convert_elem_to_ir_value(index, False), type=tl.int32)
        x = self.builder.create_get_buffer(src.handle, index.handle)
        return self.tensor(x, src.type)

    def mbar_expect(self, mbar: tl.tensor, size_in_bytes:int, pred: tl.tensor) -> TensorTy:
        # TODO: handle pred is None, then should be const True
        x = self.builder.create_mbar_expect(mbar.handle, pred.handle, size_in_bytes)
        return self.tensor(x, tl.void)

    def mbar_wait(self, mbar: tl.tensor, phase:tl.tensor) -> TensorTy:
        # phase is Value not const expr
        phase = tl.tensor(self._convert_elem_to_ir_value(phase, False), type=tl.int32)
        x = self.builder.create_mbar_wait(mbar.handle, phase.handle)
        return self.tensor(x, tl.void)

    def mbar_arrive(self, mbar: tl.tensor, pred:tl.tensor, track_async_op:bool, tx_cnt:int) -> TensorTy:
        x = self.builder.create_mbar_arrive(mbar.handle, pred.handle, track_async_op, tx_cnt)
        return self.tensor(x, tl.void)

    def _async_load_block_pointer(self, mem, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile):
        # Load by a block pointer: `pointer_type<block_type<>>`
        # Block pointer can not have `mask` and `other` arguments
        if mask is not None or other is not None:
            raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

        elt_ty = ptr.type.element_ty.element_ty
        assert elt_ty != tl.int1, "`tl.int1` should be rewritten in `tl.make_block_ptr`"
        if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
            raise ValueError("Padding option `nan` is not supported for integer block pointers")

        # `dst_ty` is de-referenced type of the pointer type
        dst_ty = ptr.type.element_ty

        # Check `boundary_check` argument
        boundary_check = self._canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())

        # Build IR
        return self.tensor(
            self.builder.create_tensor_pointer_async_load(mem.handle, ptr.handle, boundary_check, padding, cache, eviction, is_volatile),
            tl.void)

    def _async_load_legacy(self, mem, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile):
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
                mask = self.broadcast_impl_shape(mask, ptr.type.get_block_shapes())
            if other is not None:
                other = self.broadcast_impl_shape(other, ptr.type.get_block_shapes())

        # Get `pointer_type<elt_ty>` and `elt_ty`
        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        is_bool = elt_ty == tl.int1
        if is_bool:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast `other` into `elt_ty` type
        if other is not None:
            other = self.cast(other, elt_ty)

        # Create loaded result type `dst_ty`
        if ptr.type.is_block():
            dst_ty = ptr.type.with_element_ty(elt_ty)
        else:
            # Load by de-referencing the pointer of scalar
            dst_ty = elt_ty

        # Build IR
        if mask is None:
            ret = self.tensor(self.builder.create_async_load(mem.handle, ptr.handle, cache, eviction, is_volatile), tl.void)
        else:
            ret = self.tensor(
                self.builder.create_masked_async_load(mem.handle, ptr.handle, mask.handle, other.handle if other else None, cache,
                                                eviction, is_volatile), tl.void)
        #if is_bool:
        #    ret = self.cast(ret, tl.int1)
        return ret

    def async_load(self, mem: TensorTy, ptr: TensorTy, mask: Optional[TensorTy], other: Optional[TensorTy], boundary_check: Tuple,
             padding_option: str, cache_modifier: str, eviction_policy: str, is_volatile: bool) -> TensorTy:
        # Cache, eviction and padding options
        cache = self._str_to_load_cache_modifierx(cache_modifier)
        eviction = self._str_to_eviction_policyx(eviction_policy)
        padding = self._str_to_padding_optionx(padding_option)

        if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
            # Load by a block pointer: `pointer_type<block_type<>>`
            return self._async_load_block_pointer(mem, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile)
        else:
            # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
            return self._async_load_legacy(mem, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile)

    def async_load_wait(self, pendings:int) -> TensorTy:
        x = self.builder.create_async_load_wait(pendings)
        return self.tensor(x, tl.void)

    def warp_reduction(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn) -> Tuple[TensorTy, ...]:
        if axis is None:
            inputs = tuple(self.reshape(t, [t.numel.value], can_reorder=True) for t in inputs)
            axis = 0
        # get result shape
        shape = inputs[0].type.shape
        rank = len(shape)
        assert axis < rank, f"reduction axis must be < inputs rank ({rank})"
        ret_shape = [s for i, s in enumerate(shape) if i != axis]
        assert all(t.type.shape == shape for t in inputs), "all reduction inputs must have the same shape"

        reduce_op = self.builder.create_warp_reduce([t.handle for t in inputs], axis)
        region_builder_fn(reduce_op)
        assert reduce_op.verify()

        return tuple(
            self.wrap_tensor(reduce_op.get_result(i), inputs[i].type.scalar, ret_shape) for i in range(len(inputs)))
