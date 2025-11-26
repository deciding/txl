#ifndef TRITON_TXLGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TXLGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <list>
#include <vector>

namespace mlir {
namespace triton {

namespace txlgpu {

/// This does post-processing on the pipelined loop to try to pipeline wgmma
/// ops.
// TODO: this should be included as part of the pipeline but currently the wgmma
// wait modeling is problematic.
void asyncLaunchDots(scf::ForOp forOp);

/// Post process the pipelined loop by updating the wait ops with the right
/// number of groups in flight.
void updateWaits(ModuleOp module);


}; // namespace txlgpu
} // namespace triton
} // namespace mlir
#endif // TRITON_TXLGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
