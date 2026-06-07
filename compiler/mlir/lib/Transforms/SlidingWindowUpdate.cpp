#include "qfx/IR/QFXDialect.h"
#include "qfx/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace qfx {
namespace {

template <typename OpTy>
struct MarkIncrementalPattern : mlir::OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(OpTy op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (op.getIncrementalAttr())
      return mlir::failure();
    rewriter.modifyOpInPlace(op, [&] { op.setIncrementalAttr(rewriter.getUnitAttr()); });
    return mlir::success();
  }
};

struct SlidingWindowUpdatePass
    : public mlir::PassWrapper<SlidingWindowUpdatePass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SlidingWindowUpdatePass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MarkIncrementalPattern<RollingMeanOp>,
                 MarkIncrementalPattern<RollingStdOp>,
                 MarkIncrementalPattern<RollingCovOp>,
                 MarkIncrementalPattern<FusedRollingStatsOp>,
                 MarkIncrementalPattern<FusedRollingCovOp>>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  llvm::StringRef getArgument() const final { return "qfx-sliding-window"; }
  llvm::StringRef getDescription() const final {
    return "Enable trailing-window incremental updates for rolling ops";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSlidingWindowUpdatePass() {
  return std::make_unique<SlidingWindowUpdatePass>();
}

} // namespace qfx
