from __future__ import annotations
import hashlib
import json
from triton._C.libtriton import get_cache_invalidating_env_vars, ir
from triton.backends import backends
from triton.backends.compiler import Language
from triton.backends.compiler import BaseBackend, GPUTarget
from triton import __version__, knobs
from triton.runtime.autotuner import OutOfResources
from triton.runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from triton.runtime.driver import driver
from triton.tools.disasm import get_sass
from pathlib import Path
import re
import functools
import os
import sysconfig
import time

# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
# - (attributes \{[\S\s]+\})? : optionally match attributes enclosed in braces and capture it as group 3
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ptx": ptx_prototype_pattern,
}

ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ptx": ptx_arg_type_pattern,
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    tma = re.search(r'tt.nv_tma_desc = 1', x)
    if tma is not None:
        return 'nvTmaDesc'
    x = re.sub(r' {[^}]+}', '', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


class ASTSource:

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        self.fn = fn
        self.language = Language.TRITON
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = signature
        self.constants = dict()
        if constexprs is not None:
            for k, v in constexprs.items():
                k = (fn.arg_names.index(k), ) if isinstance(k, str) else k
                assert isinstance(k, tuple)
                self.constants[k] = v
        self.attrs = attrs or dict()
        if isinstance(self.signature, str):
            self.signature = {k: v.strip() for k, v in enumerate(self.signature.split(","))}
        else:
            for k in self.signature.keys():
                if not isinstance(k, str):
                    raise TypeError("Signature keys must be string")

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]
        get_key = lambda x: x.cache_key if hasattr(x, 'cache_key') else str(x)
        constants_key = '-'.join([get_key(v) for k, v in sorted(self.constants.items())])
        key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def make_ir(self, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                           module_map=module_map)

    def parse_options(self):
        return dict()


class IRSource:

    def __init__(self, path, context, backend):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.language = Language.TRITON
        self.src = path.read_text()
        ir.load_dialects(context)
        backend.load_dialects(context)

        # We don't have a easy-to-use PTX parser that we can use, so keep that regex for now.
        # TODO - replace with a proper parser
        if self.ext == "ptx":
            match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
            self.name = match.group(1)
            signature = match.group(2)
            types = re.findall(arg_type_pattern[self.ext], signature)
            self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}
        else:
            self.module = ir.parse_mlir_module(self.path, context)
            fn_name = self.module.get_entry_func_name()
            self.name = "@" + fn_name
            funcOp = self.module.get_function(fn_name)
            func_ty = self.module.get_function_signature(funcOp)
            self.signature = {k: ty for k, ty in enumerate(func_ty)}
        self.constants = None

    def hash(self):
        return hashlib.sha256(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, options, codegen_fns, module_map, context):
        self.module.context = context
        return self.module

    def parse_options(self):
        if self.ext == "ttgir":
            num_warps = self.module.get_int_attr("ttg.num-warps")
            assert num_warps is not None, "Unable to parse ttg.num-warps attribute"
            return {'num_warps': num_warps}
        return dict()


@functools.lru_cache()
def triton_key():
    import pkgutil
    # NOTE: txl
    PYTHON_TXL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TXL_PATH = os.path.dirname(os.path.dirname(PYTHON_TXL))
    TRITON_PATH = os.path.join(TXL_PATH, 'thirdparty', 'triton', 'python', 'triton')

    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]
    # compiler
    path_prefixes = [
        (os.path.join(TRITON_PATH, "compiler"), "triton.compiler."),
        (os.path.join(TRITON_PATH, "backends"), "triton.backends."),
    ]
    for path, prefix in path_prefixes:
        for lib in pkgutil.walk_packages([path], prefix=prefix):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hashlib.sha256(f.read()).hexdigest()]

    # backend
    libtriton_hash = hashlib.sha256()
    ext = sysconfig.get_config_var("EXT_SUFFIX").split(".")[-1]
    with open(os.path.join(TRITON_PATH, "_C", f"libtriton.{ext}"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.walk_packages([language_path], prefix="triton.language."):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha256(f.read()).hexdigest()]
    return f'{__version__}' + '-'.join(contents)


@functools.lru_cache()
def max_shared_mem(device):
    return driver.active.utils.get_device_properties(device)["max_shared_mem"]


def parse(full_name, ext, context):
    if ext == "ttir" or ext == "ttgir":
        module = ir.parse_mlir_module(full_name, context)
        module.context = context
        return module
    if ext == "llir" or ext == "ptx" or ext == "amdgcn":
        return Path(full_name).read_text()
    if ext == "cubin" or ext == "hsaco":
        return Path(full_name).read_bytes()

import difflib
def diff_strings_colored(str1, str2, log_dir=None, log_filename='tmp.ir'):
    if log_dir is not None:
        log_path = os.path.join(log_dir, log_filename)
        with open(log_path, 'w') as f:
            f.write(str2)
        print(f"log_dir is set, the diff is not printed, instead ir will be saved to {log_path}")
        return

    # Split the strings into lines for comparison
    lines1 = str1.splitlines()
    lines2 = str2.splitlines()

    # Create a Differ object
    differ = difflib.Differ()

    # Compute the difference between the two lists of lines
    diff = differ.compare(lines1, lines2)

    # Print the differences with colors
    for line in diff:
        if line.startswith('+'):
            print(f"\033[32m{line}\033[0m")  # Green for additions
        elif line.startswith('-'):
            print(f"\033[31m{line}\033[0m")  # Red for deletions
        elif line.startswith('?'):
            print(f"\033[34m{line}\033[0m")  # Blue for change indicators
        else:
            print(line)  # Default color for unchanged lines

def filter_traceback(e: BaseException):
    """
    Removes code_generator.py and related files from tracebacks.

    These are uninteresting to the user -- "just show me *my* code!"
    """
    if knobs.compilation.front_end_debugging:
        return

    if e.__cause__ is not None:
        filter_traceback(e.__cause__)
    if e.__context__ is not None:
        filter_traceback(e.__context__)

    # If a user has a file that matches one of these, they're out of luck.
    BAD_FILES = [
        "/triton/compiler/code_generator.py",
        "/ast.py",
    ]
    BAD_FILES = [bad_file.replace("/", os.sep) for bad_file in BAD_FILES]

    tb = e.__traceback__
    frames = []
    while tb is not None:
        if not any(f for f in BAD_FILES if tb.tb_frame.f_code.co_filename.endswith(f)):
            frames.append(tb)
        tb = tb.tb_next

    for (cur_frame, next_frame) in zip(frames, frames[1:]):
        cur_frame.tb_next = next_frame

    if not frames:
        e.__traceback__ = None
    else:
        frames[-1].tb_next = None
        e.__traceback__ = frames[0]


class CompileTimer:

    def __init__(self) -> None:
        self.start: float = time.time()
        self.ir_initialization_end: float | None = None
        self.lowering_stage_ends: list[tuple[str, float]] = []
        self.store_results_end: float | None = None

    def finished_ir_initialization(self) -> None:
        self.ir_initialization_end = time.time()

    def stage_finished(self, stage_name: str) -> None:
        self.lowering_stage_ends.append((stage_name, time.time()))

    def end(self) -> knobs.CompileTimes:
        timestamp = time.time()
        if self.ir_initialization_end is None:
            self.ir_initialization_end = timestamp
        else:
            self.store_results_end = timestamp

        def delta(start: float, end: float | None) -> int:
            if end is None:
                return 0
            return int((end - start) * 1000000)

        lowering_stage_durations = []
        stage_start = self.ir_initialization_end
        for stage_name, stage_end in self.lowering_stage_ends:
            lowering_stage_durations.append((stage_name, delta(stage_start, stage_end)))
            stage_start = stage_end

        return knobs.CompileTimes(
            ir_initialization=delta(self.start, self.ir_initialization_end),
            lowering_stages=lowering_stage_durations,
            store_results=delta(stage_start, self.store_results_end),
        )


############################################
# Start from nvidia backend compiler
############################################

from triton._C.libtriton import passes, llvm, nvidia
# fragile import
#from thirdparty.triton.third_party.nvidia.backend.compiler import get_ptx_version_from_options
def load_ptx_func(func_name):
    import importlib.util

    # Define the full path to the module
    module_path = Path(__file__).parent.parent.parent.parent / "thirdparty" / "triton" / "third_party" / "nvidia" / "backend" / "compiler.py"

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("compiler", module_path)
    compiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_module)

    # Access the function
    #get_ptx_version_from_options = compiler_module.get_ptx_version_from_options
    func = getattr(compiler_module, func_name)
    return func

def make_nv_dbg_ttir(mod, metadata, opt, capability, log_dir=None):
    pm1 = ir.pass_manager(mod.context)
    pm2 = ir.pass_manager(mod.context)

    before = mod.str()

    pm1.enable_debug()
    pm2.enable_debug()

    def add_ws_code_partition_txl(i):
        passes.ttir.add_ws_code_partition_txl(i, metadata['num_warpgroups'])

    pass_funcs = [
        passes.common.add_inliner,
        passes.ttir.add_rewrite_tensor_pointer,
        add_ws_code_partition_txl,
    ]
    # 3.4.x
    if capability // 10 < 9:
        pass_funcs += [
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer,
        ]
    pass_funcs += [
        passes.common.add_canonicalizer,
        passes.ttir.add_combine,
        passes.ttir.add_reorder_broadcast,
        passes.common.add_cse,
        passes.common.add_symbol_dce,
        passes.ttir.add_loop_unroll,
    ]

    for i, p in enumerate(pass_funcs):
        print(f"{i}. {p.__name__}")
    index = input("Choose your interest: ")
    index = int(index)
    passes1 = pass_funcs[:index]
    passes2 = pass_funcs[:index+1]

    for p in passes1:
        p(pm1)
    if index > 0:
        pm1.run(mod)
        before = mod.str()

    passes2[index](pm2)
    pm2.run(mod)

    ret = mod.str()

    diff_strings_colored(before, ret, log_dir=log_dir);

    return mod

def make_nv_dbg_ttgir(mod, metadata, opt, capability, log_dir=None, use_txl=True):
    # Set maxnreg on all kernels, if it was provided.
    if opt.maxnreg is not None:
        mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

    cluster_info = nvidia.ClusterInfo()
    if opt.cluster_dims is not None:
        cluster_info.clusterDimX = opt.cluster_dims[0]
        cluster_info.clusterDimY = opt.cluster_dims[1]
        cluster_info.clusterDimZ = opt.cluster_dims[2]
    before = mod.str()
    # TritonGPU -> LLVM-IR (MLIR)
    pm1 = ir.pass_manager(mod.context)
    pm2 = ir.pass_manager(mod.context)

    dump_enabled = pm1.enable_debug()
    dump_enabled = pm2.enable_debug()

    def add_convert_to_ttgpuir(i):
        passes.ttir.add_convert_to_ttgpuir(i, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)

    def add_plan_cta(i):
        nvidia.passes.ttnvgpuir.add_plan_cta(i, cluster_info)

    def add_smem_alloc_legalize_txl(i):
        passes.ttgpuir.add_smem_alloc_legalize_txl(i, opt.num_warps, 32, opt.num_ctas, f"cuda:{capability}")

    def add_optimize_dot_operands(i):
        passes.ttgpuir.add_optimize_dot_operands(i, capability >= 80)

    def add_optimize_dot_operands_txl(i):
        passes.ttgpuir.add_optimize_dot_operands_txl(i, capability >= 80)

    def add_hopper_warpspec(i):
        nvidia.passes.hopper.add_hopper_warpspec(i, opt.num_stages, dump_enabled)

    def add_assign_latencies(i):
        passes.ttgpuir.add_assign_latencies(i, opt.num_stages)

    def add_pipeline_txl(i):
        if use_txl:
            passes.ttgpuir.add_pipeline_txl(i, metadata['num_warpgroups'], opt.num_stages, dump_enabled)
        else:
            passes.ttgpuir.add_pipeline(i, opt.num_stages, dump_enabled)

    def add_warp_specialize(i):
        passes.ttgpuir.add_warp_specialize(i, opt.num_stages)

    # 3.3.x
    def add_ws_task_partition(i):
        passes.ttgpuir.add_ws_task_partition(i, opt.num_consumer_groups)
    def add_taskid_propagate(i):
        passes.ttgpuir.add_taskid_propagate(i, opt.num_consumer_groups)
    def add_ws_data_partition(i):
        passes.ttgpuir.add_ws_data_partition(i, opt.num_consumer_groups)
    def add_ws_code_partition(i):
        passes.ttgpuir.add_ws_code_partition(i, opt.num_buffers_warp_spec, opt.num_consumer_groups, opt.reg_dec_producer, opt.reg_inc_consumer)

    def add_ping_pong_sync(i):
        passes.ttgpuir.add_ping_pong_sync(i, opt.num_consumer_groups)
    def add_ws_lowering(i):
        passes.ttgpuir.add_ws_lowering(i, opt.num_consumer_groups)

    def add_ws_canonicalization(i):
        passes.ttgpuir.add_ws_canonicalization(i, opt.num_consumer_groups)


    pass_funcs = [
        add_convert_to_ttgpuir,
        # optimize TTGIR
        passes.ttgpuir.add_coalesce,
    ]
    if capability // 10 >= 8:
        pass_funcs += [
            passes.ttgpuir.add_f32_dot_tc
        ]
    pass_funcs += [
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        add_plan_cta,
        add_smem_alloc_legalize_txl,
        passes.ttgpuir.add_named_barrier_lower_txl,
        passes.ttgpuir.add_remove_layout_conversions,
        passes.ttgpuir.add_smem_alloc_layout_conversions_txl,
        passes.ttgpuir.add_optimize_thread_locality,
        #passes.ttgpuir.add_accelerate_matmul,
        passes.ttgpuir.add_accelerate_matmul_txl,
        passes.ttgpuir.add_remove_layout_conversions,
        #add_optimize_dot_operands,
        add_optimize_dot_operands_txl,
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding, # 3.4.x
        passes.common.add_cse,
    ]
    if capability // 10 in [8, 9]:
        pass_funcs += [
            passes.ttgpuir.add_fuse_nested_loops,
            passes.common.add_canonicalizer,
            #passes.common.add_licm, # 3.3.x
            passes.ttir.add_triton_licm, # 3.4.x
            #passes.ttgpuir.add_optimize_accumulator_init, # 3.3.x
            passes.common.add_canonicalizer,
            passes.ttgpuir.add_combine_tensor_select_and_if,
            add_hopper_warpspec, # 3.4.x
            # 3.3.x
            #add_ws_task_partition,
            #add_taskid_propagate,
            #add_ws_data_partition,
            #add_ws_code_partition,
            add_assign_latencies, # 3.4.x
            passes.ttgpuir.add_schedule_loops, # 3.4.x
            add_pipeline_txl,
            #add_ping_pong_sync, # 3.3.x
            #add_ws_lowering, # 3.3.x
        ]
    elif capability // 10 >= 10:
        pass_funcs += [
            passes.ttgpuir.add_fuse_nested_loops,
            passes.common.add_canonicalizer,
            #passes.common.add_licm, # 3.3.x
            passes.ttir.add_triton_licm, # 3.4.x
            passes.ttgpuir.add_optimize_accumulator_init,
            passes.ttgpuir.add_hoist_tmem_alloc, # 3.4.x
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem, # 3.4.x
            add_assign_latencies, # 3.4.x
            passes.ttgpuir.add_schedule_loops, # 3.4.x
            add_warp_specialize,
            # 3.3.x
            #add_ws_task_partition,
            #add_taskid_propagate,
            #add_ws_data_partition,
            #add_ws_code_partition,
            add_pipeline_txl, # TODO: for Blackwell
            passes.ttgpuir.add_combine_tensor_select_and_if,
            #nvidia.passes.ttnvgpuir.add_keep_acc_in_tmem, # 3.3.x
            #add_ws_lowering, # 3.3.x
            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens, # 3.4.x
        ]
    else:
        pass_funcs +=[
            #passes.common.add_licm, # 3.3.x
            passes.ttir.add_triton_licm, # 3.4.x
        ]
    pass_funcs += [
        passes.common.add_canonicalizer,
        passes.ttir.add_loop_aware_cse, # 3.4.x
        passes.ttgpuir.add_prefetch,
        add_optimize_dot_operands,
        passes.ttgpuir.add_coalesce_async_copy,
        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts, # 3.4.x
        passes.ttgpuir.add_remove_layout_conversions,
        nvidia.passes.ttnvgpuir.add_interleave_tmem, # 3.4.x
        passes.ttgpuir.add_reduce_data_duplication,
        passes.ttgpuir.add_reorder_instructions,
        passes.common.add_cse,
        passes.common.add_symbol_dce,
    ]
    if capability // 10 >= 9:
        pass_funcs += [
            nvidia.passes.ttnvgpuir.add_tma_lowering,
            nvidia.passes.ttnvgpuir.add_fence_insertion,
        ]
    pass_funcs += [
        passes.common.add_sccp, # 3.4.x
        passes.common.add_canonicalizer,
    ]
    #if capability // 10 >= 9: # 3.3.x
    #    pass_funcs += [
    #        add_ws_canonicalization,
    #    ]

    for i, p in enumerate(pass_funcs):
        print(f"{i}. {p.__name__}")
    index = input("Choose your interest: ")
    index = int(index)
    passes1 = pass_funcs[:index]
    passes2 = pass_funcs[:index+1]

    for p in passes1:
        p(pm1)
    if index > 0:
        pm1.run(mod)
        before = mod.str()

    passes2[index](pm2)
    pm2.run(mod)

    ret = mod.str()
    diff_strings_colored(before, ret, log_dir=log_dir);

    #pm.run(mod)
    metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
    #tensordesc_meta = mod.get_tensordesc_metadata()
    #metadata["tensordesc_meta"] = tensordesc_meta
    return mod

def make_nv_dbg_llir(backend, src, metadata, options, capability, log_dir=None):
    ptx_version = load_ptx_func('get_ptx_version_from_options')(options, backend.target.arch)
    def add_to_llvmir(i):
        return nvidia.passes.ttgpuir.add_to_llvmir(i, capability, ptx_version)


    mod = src
    before = mod.str()
    # TritonGPU -> LLVM-IR (MLIR)
    pm1 = ir.pass_manager(mod.context)
    pm2 = ir.pass_manager(mod.context)
    #pm.enable_debug()
    pass_funcs = [
        nvidia.passes.ttnvgpuir.add_lower_mma,
        passes.ttgpuir.add_combine_tensor_select_and_if,
        passes.ttgpuir.add_allocate_warp_groups,
        nvidia.passes.txlgpuir.add_txlgpu_inherit_wg_id,
        passes.convert.add_scf_to_cf,
        passes.ttgpuir.add_allocate_shared_memory,
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory,
        passes.ttgpuir.add_allocate_global_scratch_memory,
        add_to_llvmir,
        passes.common.add_canonicalizer,
        passes.common.add_cse,
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm,
        nvidia.passes.txlgpuir.add_txlgpu_to_llvm,
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm,
        passes.common.add_canonicalizer,
        passes.common.add_cse,
        passes.common.add_symbol_dce,
    ]
    if not knobs.compilation.disable_line_info:
        pass_funcs += [
            passes.llvmir.add_di_scope,
        ]

    def opt1(mod):
        llvm.init_targets()
        context = llvm.context()
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        return llvm_mod

    def opt2(llvm_mod):
        proc = load_ptx_func('sm_arch_from_capability')(capability)
        features = load_ptx_func('get_features')(options, backend.target.arch)
        triple = 'nvptx64-nvidia-cuda'
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        return llvm_mod

    def opt3(llvm_mod):
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        return llvm_mod

    opt_passes = [opt1, opt2, opt3]

    for i, p in enumerate(pass_funcs):
        print(f"{i}. {p.__name__}")
    for i, p in enumerate(opt_passes):
        print(f"{i+len(pass_funcs)}. {p.__name__}")

    index = input("Choose your interest: ")
    index = int(index)
    passes1 = pass_funcs[:index]
    passes2 = pass_funcs[:index+1]

    for p in passes1:
        p(pm1)
    if index > 0:
        pm1.run(mod)
        before = mod.str()
    if index < len(pass_funcs):
        passes2[index](pm2)
        pm2.run(mod)

    for i in range(index - len(pass_funcs) + 1):
        mod = opt_passes[i](mod)

    if index >= len(pass_funcs):
        ret = str(mod)
    else:
        ret = mod.str()
    diff_strings_colored(before, ret, log_dir=log_dir);
    # Get some metadata
    # warp-specialization mutates num_warps
    total_num_warps = src.get_int_attr("ttg.total-num-warps")
    if total_num_warps is not None:
        metadata["num_warps"] = total_num_warps
    metadata["shared"] = src.get_int_attr("ttg.shared")
    metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
    metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
    metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
    return ret

def add_dbg_stages(backend, stages, options, diff_mode='ttgir', log_dir=None, use_txl=True):
    capability = backend._parse_arch(options.arch)
    if diff_mode == 'ttir':
        stages["ttir"] = lambda src, metadata: make_nv_dbg_ttir(src, metadata, options, capability, log_dir=log_dir)
    elif diff_mode == 'ttgir':
        stages["ttir"] = lambda src, metadata: backend.make_ttir(src, metadata, options, capability)
        stages["ttgir"] = lambda src, metadata: make_nv_dbg_ttgir(src, metadata, options, capability, log_dir=log_dir, use_txl=use_txl)
    else:
        stages["ttir"] = lambda src, metadata: backend.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: backend.make_ttgir(src, metadata, options, capability, use_txl=use_txl)
        stages["llir"] = lambda src, metadata: make_nv_dbg_llir(backend, src, metadata, options, capability, log_dir=log_dir)

############################################
# End from nvidia backend compiler
############################################


def compile(src, target=None, options=None, diff_mode=None, log_dir=None, use_txl=True, txl_options=None):
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # NOTE: txl
    # create backend
    if ir_source:
        src_filename = src
        assert isinstance(src, str), "source must be either AST or a filepath"
        context = ir.context()
        src = IRSource(src, context, backend)
    else:
        signature_json = {}
        constants = repr(src.constants)
        signature = repr(src.signature)
        signature_json['constants'] = constants
        signature_json['signature'] = signature
        signature_str = json.dumps(signature_json)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars()
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = knobs.compilation.override
    enable_ir_dump = knobs.compilation.dump_ir
    store_only_binary = knobs.compilation.store_binary_only
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res

    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__
    metadata.update(txl_options) # NOTE: txl
    # run compilation pipeline  and populate metadata
    stages = dict()
    # NOTE: txl
    if diff_mode:
        add_dbg_stages(backend, stages, options, diff_mode=diff_mode, log_dir=log_dir, use_txl=use_txl)
    else:
        backend.add_stages(stages, options, src.language, use_txl=use_txl)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource):
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    # NOTE: txl
    if src.ext == 'ptx':
        module = src.src
    else:
        try:
            module = src.make_ir(options, codegen_fns, module_map, context)
        except Exception as e:
            filter_traceback(e)
            raise

    if ir_source:
        ir_filename = f"{file_name}.{src.ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    use_ir_loc = knobs.compilation.use_ir_loc
    if ir_source and use_ir_loc:
        module.create_location_snapshot(src.path)
        print(f"Creating new locations for {src.path}")

    if compilation_listener:
        timer.finished_ir_initialization()
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if fn_override_manager is None:
            # Users can override kernels at scale by setting `ir_override` in autotune config
            # without TRITON_KERNEL_OVERRIDE
            if (ir_override := metadata.get("ir_override", None)) and ir_override.endswith(f".{ext}"):
                next_module = parse(ir_override, ext, context)
        elif full_name := fn_override_manager.get_file(ir_filename):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)
    # NOTE: txl
    if diff_mode:
        return None
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # Compilation completed, disabling multithreading in context.
    # This is needed to safely finalize threads pool inside context: if current process forks before
    # python GC deletes context object, thread pool in child process will be invalid, which could
    # lead to child crash or hang.
    #
    # However disabling multithreading causes the code to hang if the ASAN pass is enabled
    # this is likely due to the llvm-symbolizer forking a process
    # TODO: Reconcile the difference here between the ASAN and non-ASAN path with enabling
    # multithreading in the MLIR context
    if not knobs.compilation.enable_asan:
        context.disable_multithreading()
    if fn_dump_manager is not None:
        fn_dump_manager.put(json.dumps(metadata, default=vars), metadata_filename)
        signature_filename = f"{file_name}_signature.json"
        fn_dump_manager.put(signature_str, signature_filename)
    # signature only comes from ASTSource
    if src.ext == 'ptx' or src.ext == 'ttgir':
        cur_path, cur_ext = os.path.splitext(src_filename)
        json_filename = f'{cur_path}.json'
        metadata_group[metadata_filename] = json_filename
        signature_filename = f'{cur_path}_signature.json'
        with open(signature_filename, "r") as file:
            signature_json = json.load(file)
        class ConstExprHandler:
            def __getitem__(self, value):
                return constexpr(value)
        const_expr_namespace = {"constexpr": ConstExprHandler()}
        src.constants = eval(signature_json['constants'], const_expr_namespace)
        src.signature = eval(signature_json['signature'])

    # notify any listener
    if compilation_listener:
        compilation_listener(src=src, metadata=metadata, metadata_group=metadata_group, times=timer.end(),
                             cache_hit=False)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)


def make_backend(target: GPUTarget) -> BaseBackend:
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
    return actives[0](target)


class LazyDict:

    def __init__(self, data):
        self.data = data
        self.extras = []

    def get(self):
        for func, args in self.extras:
            self.data = self.data | func(*args)
        self.extras.clear()
        return self.data

    def add(self, func, args):
        self.extras.append((func, args))


class AsmDict(dict):

    def __missing__(self, key):

        if key == "sass":
            value = get_sass(self["cubin"])
        else:
            raise KeyError("Unknown key: '%s'" % key)

        self[key] = value
        return value


class CompiledKernel:

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())
        metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None

    def _init_handles(self):
        if self.module is not None:
            return
        device = driver.active.get_current_device()
        # create launcher
        self.run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory")
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads")

    def __getattribute__(self, name):
        if name == 'run':
            self._init_handles()
        return super().__getattribute__(name)

    def launch_metadata(self, grid, stream, *args):
        if knobs.runtime.launch_enter_hook is None:
            return None
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_dict = {}
        arg_idx = 0
        for i, arg_name in enumerate(self.src.fn.arg_names):
            arg_dict[arg_name] = args[arg_idx]
            arg_idx += 1
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

        return runner
