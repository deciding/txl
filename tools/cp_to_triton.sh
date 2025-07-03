cp patch/triton/lib/Dialect/TritonGPU/Transforms/TaskIdPropagate.cpp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/
cp patch/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/
cp patch/triton/lib/Dialect/TritonGPU/Transforms/WSCodePartition.cpp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/
cp patch/triton/lib/Dialect/TritonGPU/Transforms/WSDataPartition.cpp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/
cp patch/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp thirdparty/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/
cp patch/triton/python/src/ir.cc thirdparty/triton/python/src/

cp patch/triton/bin/RegisterTritonDialects.h thirdparty/triton/bin
cp patch/triton/third_party/nvidia/triton_nvidia.cc thirdparty/triton/third_party/nvidia/
cp patch/triton/third_party/nvidia/backend/compiler.py thirdparty/triton/third_party/nvidia/backend/
cp patch/triton/third_party/nvidia/include/CMakeLists.txt thirdparty/triton/third_party/nvidia/include/
cp patch/triton/third_party/nvidia/include/Dialect/CMakeLists.txt thirdparty/triton/third_party/nvidia/include/Dialect/
cp patch/triton/third_party/nvidia/lib/CMakeLists.txt thirdparty/triton/third_party/nvidia/lib/
cp patch/triton/third_party/nvidia/lib/Dialect/CMakeLists.txt thirdparty/triton/third_party/nvidia/lib/Dialect/

# make sure folder no exist
#rm -rf thirdparty/triton/third_party/nvidia/include/Dialect/TXLGPU/
#rm -rf thirdparty/triton/third_party/nvidia/include/TXLGPUToLLVM/
#rm -rf thirdparty/triton/third_party/nvidia/lib/Dialect/TXLGPU/
#rm -rf thirdparty/triton/third_party/nvidia/lib/TXLGPUToLLVM/

cp -r patch/triton/third_party/nvidia/include/Dialect/TXLGPU/ thirdparty/triton/third_party/nvidia/include/Dialect/
cp -r patch/triton/third_party/nvidia/include/TXLGPUToLLVM/ thirdparty/triton/third_party/nvidia/include/
cp -r patch/triton/third_party/nvidia/lib/Dialect/TXLGPU/ thirdparty/triton/third_party/nvidia/lib/Dialect/
cp -r patch/triton/third_party/nvidia/lib/TXLGPUToLLVM/ thirdparty/triton/third_party/nvidia/lib/
