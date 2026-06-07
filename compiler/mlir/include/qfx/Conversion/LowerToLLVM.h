#pragma once

#include "qfx/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace qfx {

mlir::LogicalResult emitModuleLLVMIR(mlir::ModuleOp module, mlir::MLIRContext &context,
                                     const std::string &path);

mlir::LogicalResult emitModuleObject(mlir::ModuleOp module, mlir::MLIRContext &context,
                                     const std::string &path);

} // namespace qfx
