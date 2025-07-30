import os
import json
import hashlib
from pathlib import Path
import re

from triton import __version__
from triton.runtime.driver import driver
from triton.backends import backends
from triton.compiler.compiler import prototype_pattern, arg_type_pattern, convert_type_repr
from triton._C.libtriton import get_cache_invalidating_env_vars, ir
from triton.runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import triton_key, filter_traceback, AsmDict, LazyDict
from triton.runtime.autotuner import OutOfResources
from triton.language import constexpr

from triton.compiler.code_generator import ast_to_ttir

class ASTSource:

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        self.fn = fn
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
        return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                           module_map=module_map)

    def parse_options(self):
        return dict()

class IRSource:

    def __init__(self, path, context, backend):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
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
            # Lazy load from a signature json file
            self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}
            self.constants = None
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

def make_backend(target):
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
    return actives[0](target)

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

def make_nv_dbg_ttir(mod, metadata, opt, log_dir=None):
    pm1 = ir.pass_manager(mod.context)
    pm2 = ir.pass_manager(mod.context)

    before = mod.str()

    pm1.enable_debug()
    pm2.enable_debug()

    def add_ws_code_partition_txl(i):
        passes.ttir.add_ws_code_partition_txl(pm, metadata['num_warpgroups'])

    pass_funcs = [
        passes.common.add_inliner,
        passes.ttir.add_rewrite_tensor_pointer,
        add_ws_code_partition_txl,
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

    def add_pipeline_txl(i):
        if use_txl:
            passes.ttgpuir.add_pipeline_txl(pm, metadata['num_warpgroups'], opt.num_stages, dump_enabled)
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
        #passes.ttgpuir.add_smem_alloc_layout_conversions_txl,
        passes.ttgpuir.add_optimize_thread_locality,
        #passes.ttgpuir.add_accelerate_matmul,
        passes.ttgpuir.add_accelerate_matmul_txl,
        passes.ttgpuir.add_remove_layout_conversions,
        #add_optimize_dot_operands,
        add_optimize_dot_operands_txl,
        passes.common.add_cse,
    ]
    if capability // 10 in [8, 9]:
        pass_funcs += [
            passes.ttgpuir.add_fuse_nested_loops,
            passes.common.add_canonicalizer,
            #passes.ttir.add_triton_licm,
            passes.common.add_licm, # 3.3.x
            passes.ttgpuir.add_optimize_accumulator_init, # 3.3.x
            passes.common.add_canonicalizer,
            passes.ttgpuir.add_combine_tensor_select_and_if,
            add_ws_task_partition,
            add_taskid_propagate,
            add_ws_data_partition,
            add_ws_code_partition,
            add_pipeline_txl,
            add_ping_pong_sync,
            add_ws_lowering,
        ]
    elif capability // 10 >= 10:
        pass_funcs += [
            passes.ttgpuir.add_fuse_nested_loops,
            passes.common.add_canonicalizer,
            #passes.ttir.add_triton_licm,
            passes.common.add_licm,
            passes.ttgpuir.add_optimize_accumulator_init,
            #add_warp_specialize,
            #passes.ttgpuir.add_hoist_tmem_alloc,
            add_ws_task_partition,
            add_taskid_propagate,
            add_ws_data_partition,
            add_ws_code_partition,
            passes.ttgpuir.add_pipeline,
            passes.ttgpuir.add_combine_tensor_select_and_if,
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem,
            nvidia.passes.ttnvgpuir.add_keep_acc_in_tmem, # 3.3.x
            add_ws_lowering,
            passes.common.add_canonicalizer,
        ]
    else:
        pass_funcs +=[
            #passes.ttir.add_triton_licm,
            passes.common.add_licm,
        ]
    pass_funcs += [
        passes.ttgpuir.add_prefetch,
        add_optimize_dot_operands,
        passes.ttgpuir.add_coalesce_async_copy,
        passes.ttgpuir.add_remove_layout_conversions,
        passes.ttgpuir.add_reduce_data_duplication,
        passes.ttgpuir.add_reorder_instructions,
        passes.common.add_cse,
        passes.common.add_symbol_dce,
    ]
    if capability // 10 >= 9:
        pass_funcs += [
            nvidia.passes.ttnvgpuir.add_fence_insertion,
            nvidia.passes.ttnvgpuir.add_tma_lowering,
        ]
    pass_funcs += [
        passes.common.add_canonicalizer,
    ]
    if capability // 10 >= 9: # 3.3.x
        pass_funcs += [
            add_ws_canonicalization,
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

    #pm.run(mod)
    metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
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

    def opt1(mod):
        llvm.init_targets()
        context = llvm.context()
        if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        return llvm_mod

    def opt2(llvm_mod):
        proc = load_ptx_func('sm_arch_from_capability')(capability)
        features = load_ptx_func('get_features')(options, backend.target.arch)
        triple = 'nvptx64-nvidia-cuda'
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        nvidia.set_nvvm_reflect_ftz(llvm_mod)

        # Set maxnreg on all kernels, if it was provided.
        if options.maxnreg is not None:
            for k in llvm_mod.get_functions():
                if not k.is_declaration() and k.is_external_linkage():
                    k.set_nvvm_maxnreg(options.maxnreg)

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
    return ret

def add_dbg_stages(backend, stages, options, diff_mode='ttgir', log_dir=None, use_txl=True):
    capability = backend._parse_arch(options.arch)
    if diff_mode == 'ttir':
        stages["ttir"] = lambda src, metadata: make_nv_dbg_ttir(src, metadata, options, log_dir=log_dir)
    elif diff_mode == 'ttgir':
        stages["ttir"] = lambda src, metadata: backend.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: make_nv_dbg_ttgir(src, metadata, options, capability, log_dir=log_dir, use_txl=use_txl)
    else:
        stages["ttir"] = lambda src, metadata: backend.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: backend.make_ttgir(src, metadata, options, capability, use_txl=use_txl)
        stages["llir"] = lambda src, metadata: make_nv_dbg_llir(backend, src, metadata, options, capability, log_dir=log_dir)


def compile(src, target=None, options=None, diff_mode=None, log_dir=None, use_txl=True, txl_options=None):
    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
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
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    store_only_binary = os.environ.get("TRITON_STORE_BINARY_ONLY", "0") == "1"
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    #metadata_group_reuse = fn_dump_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
    if not always_compile and metadata_path is not None:
        # cache hit!
        return CompiledKernel(src, metadata_group, hash)
    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__
    metadata.update(txl_options)
    # run compilation pipeline  and populate metadata
    stages = dict()
    if diff_mode:
        add_dbg_stages(backend, stages, options, diff_mode=diff_mode, log_dir=log_dir, use_txl=use_txl)
    else:
        backend.add_stages(stages, options, use_txl=use_txl) #TODO: make this available for AMD
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
    if src.ext == 'ptx':
        module = src.src
    else:
        try:
            module = src.make_ir(options, codegen_fns, module_map, context)
        except Exception as e:
            filter_traceback(e)
            raise
    use_ir_loc = os.environ.get("USE_IR_LOC", None)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if (fn_override_manager is not None and (full_name := fn_override_manager.get_file(ir_filename)) is not None):
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
    if not os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
        context.disable_multithreading()
    # return handle to compiled kernel
    if diff_mode:
        return None

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

    return CompiledKernel(src, metadata_group, hash)

class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    # TODO: move out of this namespace since it's a runtime thing
    launch_enter_hook = None
    launch_exit_hook = None

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

    # run
    def _init_handles(self):
        if self.module is not None:
            return
        device = driver.active.get_current_device()
        # create launcher
        self.run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        if self.metadata.shared > max_shared:
            raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory")
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)

    def __getattribute__(self, name):
        if name == 'run':
            self._init_handles()
        return super().__getattribute__(name)

    def launch_metadata(self, grid, stream, *args):
        if CompiledKernel.launch_enter_hook is None:
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

    # This is for CompiledKernel, while jit has JITFunction with KernelInterface getitem
    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, *args)

        return runner
