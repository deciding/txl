#include <vector>
#include <string>
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "zlang_ir.h"
#include <pybind11/pybind11.h>

namespace zlang{

namespace py = pybind11;
using namespace mlir;
using namespace triton;

class ZlangBuildHandler {
    public:
        ZlangBuildHandler(TritonOpBuilder& topBuilder): builder(topBuilder) {
        }
        TritonOpBuilder& get_builder(){
            return builder;
        }
    private:
        TritonOpBuilder& builder;
};

PYBIND11_MODULE(libzlang, m) {
  m.doc() = "Python bindings to the C++ Zlang API";
  auto subm = m.def_submodule("ir");
  py::class_<ZlangBuildHandler>(subm, "handler", py::module_local(),
                            py::dynamic_attr())
    .def(py::init<TritonOpBuilder&>())
    // getters
    .def("get_dialects",
         [](ZlangBuildHandler &self, TritonOpBuilder& builder) -> std::vector<std::string>  {
             std::vector<std::string> strVec;
             for(auto d: builder.getContext()->getLoadedDialects()){
                strVec.push_back(d->getNamespace().str());
             }
             return strVec;
         })
    .def("is_registered_op",
         [](ZlangBuildHandler &self, TritonOpBuilder& builder, std::string name) -> bool {
             return builder.getContext()->isOperationRegistered(llvm::StringRef(name));
         })
    .def("is_registered_op2",
         [](ZlangBuildHandler &self, TritonOpBuilder& builder, std::string name) -> bool {
             return builder.getLastLoc().getContext()->isOperationRegistered(llvm::StringRef(name));
         })
    .def("get_any",
         [](ZlangBuildHandler &self, TritonOpBuilder* builder) {
           builder->create<mlir::gpu::BarrierOp>();
         });
  }
}
