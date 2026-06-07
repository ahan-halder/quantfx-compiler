#pragma once

#include "HIR.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace qfx {

mlir::ModuleOp lowerHirToMLIR(mlir::MLIRContext &context, const HirModule &hir);

} // namespace qfx
