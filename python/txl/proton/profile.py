from triton._C.libproton import proton as libproton
from .hook import register_txl_hook, unregister_txl_hook
from triton.profiler.flags import set_profiling_off, set_profiling_on, is_command_line
from triton.profiler.profile import _select_backend, _check_env, _get_backend_default_path, \
        activate, deactivate
from typing import Optional, Union

DEFAULT_PROFILE_NAME = "proton"
def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    hook: Optional[str] = "txl",
):
    """
    Start profiling with the given name and backend.

    Usage:

        ```python
        proton.start("my_profile")
        # do something
        proton.finalize()
        ```

    Args:
        name (str, optional): The name (with path) of the profiling session.
                              If not provided, the default name is "~/proton.hatchet".
        backend (str, optional): The backend to use for profiling.
                                 Available options are [None, "cupti", "cupti_pcsampling", "roctracer"].
                                 Defaults to None, which automatically selects the backend matching the current active runtime.
        context (str, optional): The context to use for profiling.
                                 Available options are ["shadow", "python"].
                                 Defaults to "shadow".
        data (str, optional): The data structure to use for profiling.
                              Available options are ["tree"].
                              Defaults to "tree".
        hook (str, optional): The hook to use for profiling.
                              Available options are [None, "triton"].
                              Defaults to None.
    Returns:
        session (int): The session ID of the profiling session.
    """
    if is_command_line():
        # Ignore the start() call if the script is run from the command line.
        return

    if name is None:
        name = DEFAULT_PROFILE_NAME

    if backend is None:
        backend = _select_backend()

    _check_env(backend)

    backend_path = _get_backend_default_path(backend)

    set_profiling_on()
    if hook and hook == "txl":
        register_txl_hook()
    return libproton.start(name, context, data, backend, backend_path)

def finalize(session: Optional[int] = None, output_format: str = "hatchet") -> None:
    """
    Finalizes a profiling session.
    Flush and write the profiling data to the file specified by the session name.

    Args:
        session (int, optional): The session ID to finalize. If None, all sessions are finalized. Defaults to None.
        output_format (str, optional): The output format for the profiling results.
                                       Aavailable options are ["hatchet"].

    Returns:
        None
    """
    if session is None:
        set_profiling_off()
        libproton.finalize_all(output_format)
        unregister_txl_hook()
    else:
        if is_command_line() and session != 0:
            raise ValueError("Only one session can be finalized when running from the command line.")
        libproton.finalize(session, output_format)

def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    if not isinstance(precision, list) and not isinstance(precision, tuple):
        precision = [precision]
    metric_names = ["time/ms"]
    if 'fp8' in precision:
        metric_names = ["tflop8/s"] + metric_names
    elif 'fp16' in precision:
        metric_names = ["tflop16/s"] + metric_names
    elif 'fp32' in precision:
        metric_names = ["tflop32/s"] + metric_names
    else:
        raise ValueError("the precision must be one of the following: fp8, fp16, fp32")
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)

class profile:
    """
    A context manager for entering and exiting a profile session.

    Usage:
        context manager:
        ```python
        with proton.profile("test0"):
            foo[1,](x, y)
        ```

    Args:
        name (str): The name of the profile.
    """

    def __init__(
            self,
            name: str,
            precision: Optional[Union[str, list, tuple]] = 'fp16',
            file_name: Optional[str] = None
        ):
        self.name = name
        self.precision = precision
        if file_name is None:
            file_name = name
        self.file_name = file_name
        self.prof_id = None

    def _enter_scope(self):
        self.prof_id = start(self.name)

    def _exit_scope(self):
        finalize()
        show_profile(self.precision, self.file_name)

    def __enter__(self):
        self._enter_scope()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_scope()

class record:
    """
    A context manager for entering and exiting a recording of the profile session.

    Usage:
        context manager:
        ```python
        with proton.record(prof_id):
            foo[1,](x, y)
        ```

    Args:
        prof_id (int): The profile id
    """

    def __init__(
            self,
            prof_id: Optional[int] = 0,
        ):
        self.prof_id = prof_id

    def __enter__(self):
        activate(self.prof_id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        deactivate(self.prof_id)

