import re
from typing import Any
from triton.language.core import constexpr
from triton.language import tensor

def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v

def _is_triton_tensor(o: Any) -> bool:
    return isinstance(o, tensor)

def _apply_binary_method(method_name, lhs, rhs, _builder=None):
    if _is_triton_tensor(lhs):
        return getattr(lhs, method_name)(rhs, _builder=_builder)
    if _is_triton_tensor(rhs):
        reverse_method_name = re.sub('__(.*)__', '__r\\1__', method_name)
        return getattr(rhs, reverse_method_name)(lhs, _builder=_builder)
    return getattr(lhs, method_name)(rhs)