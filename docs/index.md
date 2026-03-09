# TeraXLang (formerly Triton Xtra Language)

TeraXLang is a Triton-based language extension that provides additional features for GPU programming, including:

- **Shared Memory Operations**: Efficient smem/tmem operations with fragment-based loading/storing
- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated memory operations
- **Warp-level Primitives**: Warp-level reductions and synchronization
- **Mbars (Mailboxes)**: Fine-grained thread synchronization

## API Reference

- [Core API](api/core.md) - Low-level TeraXLang primitives
- [JIT API](api/jit.md) - JIT compilation utilities
