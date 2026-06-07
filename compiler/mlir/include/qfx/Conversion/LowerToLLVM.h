#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace qfx {

mlir::LogicalResult lowerModuleToLLVM(mlir::ModuleOp module,
                                      mlir::MLIRContext &context);

mlir::LogicalResult emitModuleLLVMIR(mlir::ModuleOp module, mlir::MLIRContext &context,
                                     const std::string &path);

mlir::LogicalResult emitModuleObject(mlir::ModuleOp module, mlir::MLIRContext &context,
                                     const std::string &path);

} // namespace qfx
