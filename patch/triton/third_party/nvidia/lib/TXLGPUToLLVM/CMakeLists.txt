add_triton_library(TXLGPUToLLVM
    TXLGPUToLLVMPass.cpp

    DEPENDS
    TXLGPUConversionPassIncGen
    TXLGPUIR
)
