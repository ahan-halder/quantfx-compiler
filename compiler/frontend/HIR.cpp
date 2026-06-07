#include "HIR.h"

#include <sstream>
#include <unordered_map>

namespace qfx {

namespace {

HirOp parseOpName(const std::string &name, std::string &error) {
  if (name == "rolling_mean")
    return HirOp::RollingMean;
  if (name == "rolling_std")
    return HirOp::RollingStd;
  if (name == "rolling_vwap")
    return HirOp::RollingVwap;
  if (name == "rolling_cov")
    return HirOp::RollingCov;
  if (name == "zscore")
    return HirOp::ZScore;
  if (name == "ema")
    return HirOp::Ema;
  if (name == "crossover")
    return HirOp::Crossover;

  std::ostringstream oss;
  oss << "unknown operation '" << name << "'";
  error = oss.str();
  return HirOp::Variable;
}

int64_t findWindow(const Expr &expr, std::string &error) {
  for (const auto &arg : expr.namedArgs) {
    if (arg.name == "window")
      return arg.intValue;
  }
  error = "missing required named argument 'window'";
  return 0;
}

double findAlpha(const Expr &expr, std::string &error) {
  for (const auto &arg : expr.namedArgs) {
    if (arg.name == "alpha")
      return arg.isFloat ? arg.floatValue : static_cast<double>(arg.intValue);
  }
  error = "missing required named argument 'alpha'";
  return 0.0;
}

} // namespace

HirModule lowerToHir(const Module &module, std::string &error) {
  HirModule hir;
  hir.imports = module.imports;
  hir.emits = module.emits;

  std::unordered_map<std::string, Type> types;
  for (const auto &import : module.imports)
    types.emplace(import.name, import.type);

  for (const auto &let : module.lets) {
    HirValue value;
    value.name = let.name;

    if (let.expr.kind == Expr::Kind::Variable) {
      auto it = types.find(let.expr.name);
      if (it == types.end()) {
        error = "unknown variable '" + let.expr.name + "'";
        return {};
      }
      value.op = HirOp::Variable;
      value.type = it->second;
      value.operands.push_back(let.expr.name);
    } else {
      value.op = parseOpName(let.expr.name, error);
      if (!error.empty())
        return {};

      switch (value.op) {
      case HirOp::RollingMean:
      case HirOp::RollingStd:
      case HirOp::RollingCov: {
        if (let.expr.positionalArgs.empty()) {
          error = let.expr.name + " requires at least one operand";
          return {};
        }
        value.window = findWindow(let.expr, error);
        if (!error.empty())
          return {};
        value.operands = let.expr.positionalArgs;
        auto it = types.find(let.expr.positionalArgs.front());
        if (it == types.end()) {
          error = "unknown stream '" + let.expr.positionalArgs.front() + "'";
          return {};
        }
        value.type = it->second;
        break;
      }
      case HirOp::RollingVwap: {
        if (let.expr.positionalArgs.size() < 2) {
          error = "rolling_vwap requires prices and volumes operands";
          return {};
        }
        value.window = findWindow(let.expr, error);
        if (!error.empty())
          return {};
        value.operands = let.expr.positionalArgs;
        value.type = types.at(let.expr.positionalArgs.front());
        break;
      }
      case HirOp::ZScore: {
        if (let.expr.positionalArgs.size() < 3) {
          error = "zscore requires x, mu, and sigma operands";
          return {};
        }
        value.operands = let.expr.positionalArgs;
        value.type = types.at(let.expr.positionalArgs.front());
        break;
      }
      case HirOp::Ema: {
        if (let.expr.positionalArgs.empty()) {
          error = "ema requires one operand";
          return {};
        }
        value.alpha = findAlpha(let.expr, error);
        if (!error.empty())
          return {};
        value.operands = let.expr.positionalArgs;
        value.type = types.at(let.expr.positionalArgs.front());
        break;
      }
      case HirOp::Crossover: {
        if (let.expr.positionalArgs.size() < 2) {
          error = "crossover requires two operands";
          return {};
        }
        value.operands = let.expr.positionalArgs;
        value.type = types.at(let.expr.positionalArgs.front());
        break;
      }
      default:
        error = "unsupported expression";
        return {};
      }
    }

    types.emplace(value.name, value.type);
    hir.bindings.push_back(std::move(value));
  }

  for (const auto &emit : module.emits) {
    if (!types.contains(emit.name)) {
      error = "unknown emit target '" + emit.name + "'";
      return {};
    }
  }

  return hir;
}

} // namespace qfx
