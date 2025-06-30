from triton.profiler.state import enter_state, exit_state
from triton.profiler.scope import enter_scope, exit_scope
from ..compiler.compiler import CompiledKernel, LazyDict

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TXLHook:
    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        metadata = lazy_dict.get()
        exit_state()
        fn_metrics = {k: metadata[k] for k in TXLHook.metrics if k in metadata}
        enter_scope(metadata["name"], triton_op=True, metrics=fn_metrics)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        exit_scope(triton_op=True)


def register_txl_hook() -> None:
    if CompiledKernel.launch_enter_hook is None:
        CompiledKernel.launch_enter_hook = TXLHook.enter
        CompiledKernel.launch_exit_hook = TXLHook.exit


def unregister_txl_hook() -> None:
    if CompiledKernel.launch_enter_hook == TXLHook.enter:
        CompiledKernel.launch_enter_hook = None
        CompiledKernel.launch_exit_hook = None
