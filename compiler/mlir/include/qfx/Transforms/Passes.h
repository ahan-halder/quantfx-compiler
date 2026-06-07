#pragma once

#include "mlir/Pass/Pass.h"

namespace qfx {

enum class OptLevel { O0, O1, O2, O3 };

void registerQFXPasses();

std::unique_ptr<mlir::Pass> createWindowFusionPass();
std::unique_ptr<mlir::Pass> createSlidingWindowUpdatePass();
std::unique_ptr<mlir::Pass> createLoopTilingPass(int64_t tileSize = 64);
std::unique_ptr<mlir::Pass> createVectorizeLoopsPass(int64_t vectorWidth = 8);
std::unique_ptr<mlir::Pass> createPrefetchInsertionPass(int64_t distance = 8);

mlir::LogicalResult runCPUPipeline(mlir::ModuleOp module, mlir::MLIRContext &context,
                                   OptLevel level, bool streaming);

} // namespace qfx
