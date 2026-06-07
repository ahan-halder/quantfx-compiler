#include "Benchmark.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace qfx {

namespace {

llvm::CodeGenOptLevel mapCodegenOpt(int level) {
  switch (level) {
  case 0:
    return llvm::CodeGenOptLevel::None;
  case 1:
    return llvm::CodeGenOptLevel::Less;
  case 2:
    return llvm::CodeGenOptLevel::Default;
  default:
    return llvm::CodeGenOptLevel::Aggressive;
  }
}

BenchStats computeStats(std::vector<double> samplesUs) {
  BenchStats stats;
  if (samplesUs.empty())
    return stats;
  std::sort(samplesUs.begin(), samplesUs.end());
  stats.medianUs = samplesUs[samplesUs.size() / 2];
  stats.meanUs =
      std::accumulate(samplesUs.begin(), samplesUs.end(), 0.0) /
      static_cast<double>(samplesUs.size());
  const size_t p99Index =
      std::min(samplesUs.size() - 1, static_cast<size_t>(std::ceil(0.99 * samplesUs.size()) - 1));
  stats.p99Us = samplesUs[p99Index];
  return stats;
}

} // namespace

bool runBenchmark(mlir::ModuleOp module, const HirModule &hir, const CompilerConfig &config,
                  int warmupIters, int measureIters, BenchStats &stats) {
  mlir::ExecutionEngineOptions options;
  options.jitCodeGenOptLevel = mapCodegenOpt(config.llvmCodegenOpt);
  auto engine = mlir::ExecutionEngine::create(module, options);
  if (!engine)
    return false;

  const int64_t n = hir.imports.front().type.length;
  std::vector<float> prices(n), volumes(n);
  for (int64_t i = 0; i < n; ++i) {
    prices[i] = static_cast<float>(i % 97) * 0.01f - 0.5f;
    volumes[i] = 1.0f + static_cast<float>(i % 7) * 0.1f;
  }

  std::unordered_map<std::string, std::vector<float> *> streams = {
      {"prices", &prices},
      {"prices_a", &prices},
      {"prices_b", &volumes},
      {"volumes", &volumes},
  };

  std::vector<std::vector<float>> outputs(hir.emits.size(), std::vector<float>(n, 0.0f));

  auto makeRef = [&](std::vector<float> &data) {
    StridedMemRefType<float, 1> ref;
    ref.basePtr = data.data();
    ref.data = data.data();
    ref.offset = 0;
    ref.sizes[0] = n;
    ref.strides[0] = 1;
    return ref;
  };

  std::vector<StridedMemRefType<float, 1>> importRefs;
  for (auto &import : hir.imports) {
    auto it = streams.find(import.name);
    if (it == streams.end())
      return false;
    importRefs.push_back(makeRef(*it->second));
  }

  std::vector<StridedMemRefType<float, 1>> outputRefs;
  for (auto &out : outputs)
    outputRefs.push_back(makeRef(out));

  std::vector<void *> args;
  for (auto &ref : importRefs)
    args.push_back(&ref);
  for (auto &ref : outputRefs)
    args.push_back(&ref);

  auto invokeOnce = [&]() -> bool {
    return mlir::succeeded((*engine)->invoke("qfx_kernel", args));
  };

  for (int i = 0; i < warmupIters; ++i) {
    if (!invokeOnce())
      return false;
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(measureIters));
  for (int i = 0; i < measureIters; ++i) {
    auto start = std::chrono::steady_clock::now();
    if (!invokeOnce())
      return false;
    auto end = std::chrono::steady_clock::now();
    double us =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
    samples.push_back(us);
  }

  stats = computeStats(std::move(samples));
  return true;
}

void printBenchmarkJson(const BenchStats &stats, int warmup, int iters) {
  std::cout << "{"
            << "\"median_us\":" << stats.medianUs << ","
            << "\"p99_us\":" << stats.p99Us << ","
            << "\"mean_us\":" << stats.meanUs << ","
            << "\"warmup\":" << warmup << ","
            << "\"iters\":" << iters << "}\n";
}

} // namespace qfx
