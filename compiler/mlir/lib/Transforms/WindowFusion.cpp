#include "qfx/IR/QFXDialect.h"
#include "qfx/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace qfx {
namespace {

struct FuseMeanStdPattern : mlir::OpRewritePattern<RollingMeanOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(RollingMeanOp meanOp,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Value input = meanOp.getInput();
    int64_t window = meanOp.getWindow();

    RollingStdOp stdOp = nullptr;
    for (auto *user : input.getUsers()) {
      if (user == meanOp.getOperation())
        continue;
      if (auto candidate = llvm::dyn_cast<RollingStdOp>(user)) {
        if (candidate.getWindow() == window && stdOp == nullptr)
          stdOp = candidate;
      }
    }
    if (!stdOp)
      return mlir::failure();

    auto type = llvm::cast<mlir::MemRefType>(input.getType());
    auto loc = meanOp.getLoc();
    auto fused = rewriter.create<FusedRollingStatsOp>(
        loc, mlir::TypeRange{type, type}, input, rewriter.getI64IntegerAttr(window),
        mlir::UnitAttr());
    rewriter.replaceOp(meanOp, fused.getMean());
    rewriter.replaceOp(stdOp, fused.getStddev());
    return mlir::success();
  }
};

struct FuseRollingCovPattern : mlir::OpRewritePattern<RollingCovOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(RollingCovOp covOp,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = covOp.getLoc();
    auto fused = rewriter.create<FusedRollingCovOp>(
        loc, covOp.getType(), covOp.getLhs(), covOp.getRhs(),
        rewriter.getI64IntegerAttr(covOp.getWindow()), mlir::UnitAttr());
    rewriter.replaceOp(covOp, fused.getCov());
    return mlir::success();
  }
};

struct WindowFusionPass
    : public mlir::PassWrapper<WindowFusionPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WindowFusionPass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuseMeanStdPattern, FuseRollingCovPattern>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  llvm::StringRef getArgument() const final { return "qfx-window-fusion"; }
  llvm::StringRef getDescription() const final {
    return "Fuse adjacent rolling ops over the same input/window";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createWindowFusionPass() {
  return std::make_unique<WindowFusionPass>();
}

} // namespace qfx
