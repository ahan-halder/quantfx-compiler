#pragma once

#include "HIR.h"

#include "mlir/Support/LogicalResult.h"

#include <string>

namespace qfx {

std::string generateCudaSource(const HirModule &hir);

mlir::LogicalResult emitCudaKernel(const HirModule &hir, const std::string &outputPath,
                                   bool emitPtx);

} // namespace qfx
