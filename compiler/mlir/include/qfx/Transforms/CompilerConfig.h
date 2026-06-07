#pragma once

#include <cstdint>

namespace qfx {

enum class OptLevel { O0, O1, O2, O3 };

struct CompilerConfig {
  OptLevel level = OptLevel::O2;
  bool streaming = false;
  bool windowFusion = true;
  bool loopTiling = true;
  bool prefetch = true;
  bool vectorize = false;
  int64_t tileSize = 64;
  int64_t vectorWidth = 16;
  int64_t prefetchDistance = 8;
  int llvmCodegenOpt = 3;
};

} // namespace qfx
