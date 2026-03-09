# JIT API

## `@jit` Decorator

The `@jit` decorator compiles Triton kernels with TeraXLang extensions.

```python
from teraxlang import jit

@jit
def kernel(X, Y, Z, **kwargs):
    # Your kernel code here
    pass
```

## `Config`

Configuration for kernel autotuning.

```python
from teraxlang import Config

config = Config(
    {"BLOCK_M": 128, "BLOCK_N": 256},
    {"BLOCK_M": 256, "BLOCK_N": 128},
)
```

## `@autotune` Decorator

Automatically tune kernel configurations.

```python
from teraxlang import autotune, Config

@autotune(
    configs=[
        Config({"BLOCK_M": 128, "BLOCK_N": 256}),
        Config({"BLOCK_M": 256, "BLOCK_N": 128}),
    ],
    key="M * N",
)
def kernel(X, Y, Z, **kwargs):
    ...
```

## `@heuristics` Decorator

Apply heuristics to choose configurations.

```python
from teraxlang import heuristics

@heuristics({
    "BLOCK_M": lambda M: 128 if M < 1024 else 256,
})
def kernel(X, Y, Z, **kwargs):
    ...
```
