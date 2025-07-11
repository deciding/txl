cp thirdparty/triton/bin/RegisterTritonDialects.h patch/triton/bin

cp thirdparty/triton/include/CMakeLists.txt patch/triton/include/
cp thirdparty/triton/lib/Dialect/CMakeLists.txt patch/triton/lib/Dialect/

cp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/TaskIdPropagate.cpp patch/triton/lib/Dialect/TritonGPU/Transforms/
cp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp patch/triton/lib/Dialect/TritonGPU/Transforms/
cp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/WSCodePartition.cpp patch/triton/lib/Dialect/TritonGPU/Transforms/
cp thirdparty/triton/lib/Dialect/TritonGPU/Transforms/WSDataPartition.cpp patch/triton/lib/Dialect/TritonGPU/Transforms/

cp thirdparty/triton/python/src/ir.cc patch/triton/python/src/
cp thirdparty/triton/python/src/passes.cc patch/triton/python/src/

cp thirdparty/triton/third_party/nvidia/backend/compiler.py patch/triton/third_party/nvidia/backend/

cp thirdparty/triton/third_party/nvidia/include/CMakeLists.txt patch/triton/third_party/nvidia/include/
cp thirdparty/triton/third_party/nvidia/include/Dialect/CMakeLists.txt patch/triton/third_party/nvidia/include/Dialect/
cp thirdparty/triton/third_party/nvidia/lib/CMakeLists.txt patch/triton/third_party/nvidia/lib/
cp thirdparty/triton/third_party/nvidia/lib/Dialect/CMakeLists.txt patch/triton/third_party/nvidia/lib/Dialect/
cp thirdparty/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp patch/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/

cp thirdparty/triton/third_party/nvidia/triton_nvidia.cc patch/triton/third_party/nvidia/

# make sure folder no exist
#rm -rf patch/triton/include/txl
#rm -rf patch/triton/lib/Dialect/TXL/
#rm -rf patch/triton/third_party/nvidia/include/Dialect/TXLGPU/
#rm -rf patch/triton/third_party/nvidia/include/TXLGPUToLLVM/
#rm -rf patch/triton/third_party/nvidia/lib/Dialect/TXLGPU/
#rm -rf patch/triton/third_party/nvidia/lib/TXLGPUToLLVM/
cp -r thirdparty/triton/include/txl/ patch/triton/include/
cp -r thirdparty/triton/lib/Dialect/TXL/ patch/triton/lib/Dialect/
cp -r thirdparty/triton/third_party/nvidia/include/Dialect/TXLGPU/ patch/triton/third_party/nvidia/include/Dialect/
cp -r thirdparty/triton/third_party/nvidia/include/TXLGPUToLLVM/ patch/triton/third_party/nvidia/include/
cp -r thirdparty/triton/third_party/nvidia/lib/Dialect/TXLGPU/ patch/triton/third_party/nvidia/lib/Dialect/
cp -r thirdparty/triton/third_party/nvidia/lib/TXLGPUToLLVM/ patch/triton/third_party/nvidia/lib/
