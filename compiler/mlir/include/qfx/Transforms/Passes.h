#pragma once

#include "qfx/Transforms/CompilerConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace qfx {

void registerQFXPasses();

std::unique_ptr<mlir::Pass> createWindowFusionPass();
std::unique_ptr<mlir::Pass> createSlidingWindowUpdatePass();
std::unique_ptr<mlir::Pass> createLoopTilingPass(int64_t tileSize = 64);
std::unique_ptr<mlir::Pass> createVectorizeLoopsPass(int64_t vectorWidth = 8);
std::unique_ptr<mlir::Pass> createPrefetchInsertionPass(int64_t distance = 8);

mlir::LogicalResult runCPUPipeline(mlir::ModuleOp module, mlir::MLIRContext &context,
                                   const CompilerConfig &config);

} // namespace qfx
