#pragma once

#include "AST.h"

#include <map>
#include <string>
#include <vector>

namespace qfx {

enum class HirOp {
  RollingMean,
  RollingStd,
  RollingVwap,
  RollingCov,
  ZScore,
  Ema,
  Crossover,
  Variable,
};

struct HirValue {
  std::string name;
  HirOp op = HirOp::Variable;
  Type type{};
  std::vector<std::string> operands;
  int64_t window = 0;
  double alpha = 0.0;
};

struct HirModule {
  std::vector<ImportStmt> imports;
  std::vector<HirValue> bindings;
  std::vector<EmitStmt> emits;
};

HirModule lowerToHir(const Module &module, std::string &error);

} // namespace qfx
