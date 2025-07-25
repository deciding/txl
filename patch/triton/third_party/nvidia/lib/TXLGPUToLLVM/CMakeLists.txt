add_triton_library(TXLGPUToLLVM
    TXLGPUToLLVMPass.cpp
    InheritWGIdPass.cpp

    DEPENDS
    TXLGPUConversionPassIncGen
    TXLGPUIR
)
