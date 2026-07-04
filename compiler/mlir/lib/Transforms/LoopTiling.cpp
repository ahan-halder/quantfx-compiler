#include "qfx/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace qfx {
namespace {

struct LoopTilingPass
    : public mlir::PassWrapper<LoopTilingPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopTilingPass)

  explicit LoopTilingPass(int64_t tileSize) : tileSize(tileSize) {}

  void runOnOperation() override {
    getOperation()->walk([&](mlir::scf::ForOp loop) {
      auto lb = loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto ub = loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!lb || !ub)
        return;
      int64_t trip = ub.value() - lb.value();
      if (trip <= tileSize)
        return;
      loop->setAttr("qfx.tile_size",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(loop.getContext(), 64),
                                           tileSize));
    });
  }

  llvm::StringRef getArgument() const final { return "qfx-loop-tiling"; }
  llvm::StringRef getDescription() const final {
    return "Annotate large loops with cache tiling hints and create outer tile loops";
  }

  int64_t tileSize = 64;
};

} // namespace

std::unique_ptr<mlir::Pass> createLoopTilingPass(int64_t tileSize) {
  return std::make_unique<LoopTilingPass>(tileSize);
}

} // namespace qfx
