TO_DIR=thirdparty/triton
FROM_DIR=patch/triton

cp ${FROM_DIR}/bin/RegisterTritonDialects.h ${TO_DIR}/bin

cp ${FROM_DIR}/include/CMakeLists.txt ${TO_DIR}/include/

cp ${FROM_DIR}/include/triton/Analysis/Allocation.h ${TO_DIR}/include/triton/Analysis/

cp ${FROM_DIR}/include/triton/Analysis/TXLUtility.h ${TO_DIR}/include/triton/Analysis/

cp ${FROM_DIR}/include/triton/Conversion/TritonToTritonGPU/Passes.td ${TO_DIR}/include/triton/Conversion/TritonToTritonGPU/

#include/txl

cp ${FROM_DIR}/lib/Analysis/Allocation.cpp ${TO_DIR}/lib/Analysis/
cp ${FROM_DIR}/lib/Analysis/AxisInfo.cpp ${TO_DIR}/lib/Analysis/
cp ${FROM_DIR}/lib/Analysis/CMakeLists.txt ${TO_DIR}/lib/Analysis/
cp ${FROM_DIR}/lib/Analysis/Membar.cpp ${TO_DIR}/lib/Analysis/
cp ${FROM_DIR}/lib/Analysis/TXLUtility.cpp ${TO_DIR}/lib/Analysis/

cp ${FROM_DIR}/lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp ${TO_DIR}/lib/Conversion/TritonGPUToLLVM/
cp ${FROM_DIR}/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp ${TO_DIR}/lib/Conversion/TritonToTritonGPU/
cp ${FROM_DIR}/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp ${TO_DIR}/lib/Conversion/TritonToTritonGPU/

cp ${FROM_DIR}/lib/Dialect/CMakeLists.txt ${TO_DIR}/lib/Dialect/

#lib/Dialect/TXL

cp ${FROM_DIR}/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp ${TO_DIR}/lib/Dialect/Triton/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp ${TO_DIR}/lib/Dialect/TritonGPU/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp ${TO_DIR}/lib/Dialect/TritonGPU/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp ${TO_DIR}/lib/Dialect/TritonGPU/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp ${TO_DIR}/lib/Dialect/TritonGPU/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonGPU/Transforms/Utility.cpp ${TO_DIR}/lib/Dialect/TritonGPU/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeLists.txt ${TO_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/
cp ${FROM_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp ${TO_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/

cp ${FROM_DIR}/python/src/ir.cc ${TO_DIR}/python/src/
cp ${FROM_DIR}/python/src/passes.cc ${TO_DIR}/python/src/

cp ${FROM_DIR}/third_party/nvidia/backend/compiler.py ${TO_DIR}/third_party/nvidia/backend/

cp ${FROM_DIR}/third_party/nvidia/include/CMakeLists.txt ${TO_DIR}/third_party/nvidia/include/
cp ${FROM_DIR}/third_party/nvidia/include/Dialect/CMakeLists.txt ${TO_DIR}/third_party/nvidia/include/Dialect/

#third_party/nvidia/include/Dialect/TXLGPU
#third_party/nvidia/include/TXLGPUToLLVM

cp ${FROM_DIR}/third_party/nvidia/lib/CMakeLists.txt ${TO_DIR}/third_party/nvidia/lib/
cp ${FROM_DIR}/third_party/nvidia/lib/Dialect/CMakeLists.txt ${TO_DIR}/third_party/nvidia/lib/Dialect/

#third_party/nvidia/lib/Dialect/TXLGPU
#third_party/nvidia/lib/TXLGPUToLLVM

cp ${FROM_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp ${TO_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/
cp ${FROM_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp ${TO_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/

cp ${FROM_DIR}/third_party/nvidia/triton_nvidia.cc ${TO_DIR}/third_party/nvidia/

# make sure folder no exist
rm -rf ${TO_DIR}/include/txl
rm -rf ${TO_DIR}/lib/Dialect/TXL/
rm -rf ${TO_DIR}/third_party/nvidia/include/Dialect/TXLGPU/
rm -rf ${TO_DIR}/third_party/nvidia/include/TXLGPUToLLVM/
rm -rf ${TO_DIR}/third_party/nvidia/lib/Dialect/TXLGPU/
rm -rf ${TO_DIR}/third_party/nvidia/lib/TXLGPUToLLVM/
cp -r ${FROM_DIR}/include/txl/ ${TO_DIR}/include/
cp -r ${FROM_DIR}/lib/Dialect/TXL/ ${TO_DIR}/lib/Dialect/
cp -r ${FROM_DIR}/third_party/nvidia/include/Dialect/TXLGPU/ ${TO_DIR}/third_party/nvidia/include/Dialect/
cp -r ${FROM_DIR}/third_party/nvidia/include/TXLGPUToLLVM/ ${TO_DIR}/third_party/nvidia/include/
cp -r ${FROM_DIR}/third_party/nvidia/lib/Dialect/TXLGPU/ ${TO_DIR}/third_party/nvidia/lib/Dialect/
cp -r ${FROM_DIR}/third_party/nvidia/lib/TXLGPUToLLVM/ ${TO_DIR}/third_party/nvidia/lib/

find patch/triton -type f | wc -l
