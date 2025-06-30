// HEADERS
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

// PATCH: py::class_<TritonOpBuilder>(m, "builder"
    .def("get_dialects",
         [](TritonOpBuilder& self) -> std::vector<std::string>  {
             std::vector<std::string> strVec;
             for(auto d: self.getContext()->getLoadedDialects()){
                strVec.push_back(d->getNamespace().str());
             }
             return strVec;
         })
    .def("is_registered_op",
         [](TritonOpBuilder& self, std::string name) -> bool {
             return builder.getContext()->isOperationRegistered(llvm::StringRef(name));
         })
    .def("threadidx_x",
         [](TritonOpBuilder& self) {
           self.create<mlir::gpu::ThreadIdOp>(0);
         });

