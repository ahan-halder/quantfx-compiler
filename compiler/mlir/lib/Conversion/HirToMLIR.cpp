#include "qfx/Conversion/HirToMLIR.h"

#include "HIR.h"
#include "qfx/IR/QFXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace qfx {

namespace {

mlir::MemRefType streamMemRefType(const Type &type, mlir::MLIRContext &ctx) {
  return mlir::MemRefType::get({type.length}, mlir::Float32Type::get(&ctx));
}

} // namespace

mlir::ModuleOp lowerHirToMLIR(mlir::MLIRContext &context, const HirModule &hir) {
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = builder.create<mlir::ModuleOp>(loc);

  if (hir.imports.empty())
    return module;

  const int64_t length = hir.imports.front().type.length;
  auto f32Type = mlir::Float32Type::get(&context);
  auto memrefType = mlir::MemRefType::get({length}, f32Type);

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> resultTypes;
  llvm::SmallVector<std::string> importNames;
  llvm::SmallVector<std::string> outputNames;

  for (const auto &import : hir.imports) {
    argTypes.push_back(streamMemRefType(import.type, context));
    importNames.push_back(import.name);
  }
  for (const auto &emit : hir.emits) {
    argTypes.push_back(streamMemRefType(emit.type, context));
    outputNames.push_back(emit.name);
  }

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto func = builder.create<mlir::func::FuncOp>(loc, "qfx_kernel", funcType);
  func->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                mlir::UnitAttr::get(&context));
  module.push_back(func);

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::DenseMap<llvm::StringRef, mlir::Value> env;
  for (size_t i = 0; i < importNames.size(); ++i)
    env[importNames[i]] = entry->getArgument(static_cast<unsigned>(i));

  auto lookup = [&](const std::string &name) -> mlir::Value {
    return env[name];
  };

  for (const auto &binding : hir.bindings) {
    mlir::Value result;
    switch (binding.op) {
    case HirOp::Variable:
      result = lookup(binding.operands.front());
      break;
    case HirOp::RollingMean:
      result = builder.create<RollingMeanOp>(
          loc, memrefType, lookup(binding.operands[0]),
          builder.getI64IntegerAttr(binding.window), mlir::UnitAttr());
      break;
    case HirOp::RollingStd:
      result = builder.create<RollingStdOp>(
          loc, memrefType, lookup(binding.operands[0]),
          builder.getI64IntegerAttr(binding.window), mlir::UnitAttr());
      break;
    case HirOp::RollingVwap:
      result = builder.create<RollingVwapOp>(
          loc, memrefType, lookup(binding.operands[0]),
          lookup(binding.operands[1]),
          builder.getI64IntegerAttr(binding.window));
      break;
    case HirOp::RollingCov:
      result = builder.create<RollingCovOp>(
          loc, memrefType, lookup(binding.operands[0]),
          lookup(binding.operands[1]),
          builder.getI64IntegerAttr(binding.window), mlir::UnitAttr());
      break;
    case HirOp::ZScore:
      result = builder.create<ZScoreOp>(
          loc, memrefType, lookup(binding.operands[0]),
          lookup(binding.operands[1]), lookup(binding.operands[2]));
      break;
    case HirOp::Ema:
      result = builder.create<EmaOp>(loc, memrefType, lookup(binding.operands[0]),
                                     builder.getF64FloatAttr(binding.alpha));
      break;
    case HirOp::Crossover:
      result = builder.create<CrossoverOp>(loc, memrefType,
                                           lookup(binding.operands[0]),
                                           lookup(binding.operands[1]));
      break;
    }
    env[binding.name] = result;
  }

  for (size_t i = 0; i < hir.emits.size(); ++i) {
    const auto &emit = hir.emits[i];
    mlir::Value value = env[emit.name];
    mlir::Value output =
        entry->getArgument(static_cast<unsigned>(importNames.size() + i));
    builder.create<EmitOp>(loc, value, output);
  }

  builder.create<mlir::func::ReturnOp>(loc);
  return module;
}

} // namespace qfx
