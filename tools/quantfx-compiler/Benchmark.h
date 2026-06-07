#pragma once

#include "HIR.h"
#include "qfx/Transforms/CompilerConfig.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace qfx {

struct BenchStats {
  double medianUs = 0.0;
  double p99Us = 0.0;
  double meanUs = 0.0;
};

bool runBenchmark(mlir::ModuleOp module, const HirModule &hir, const CompilerConfig &config,
                  int warmupIters, int measureIters, BenchStats &stats);

void printBenchmarkJson(const BenchStats &stats, int warmup, int iters);

} // namespace qfx
