#include "qfx/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace qfx {
namespace {

struct VectorizeSimpleLoopPattern : mlir::OpRewritePattern<mlir::scf::ForOp> {
  explicit VectorizeSimpleLoopPattern(mlir::MLIRContext *ctx, int64_t width)
      : OpRewritePattern(ctx), vectorWidth(width) {}

  mlir::LogicalResult matchAndRewrite(mlir::scf::ForOp loop,
                                      mlir::PatternRewriter &rewriter) const override {
    if (loop->hasAttr("qfx.vectorized"))
      return mlir::failure();

    if (!loop.getInitArgs().empty())
      return mlir::failure();

    auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!step || step.value() != 1)
      return mlir::failure();

    auto &body = loop.getBody()->getOperations();
    if (body.size() != 2)
      return mlir::failure();

    auto storeOp = llvm::dyn_cast<mlir::memref::StoreOp>(&body.front());
    auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(&body.back());
    if (!storeOp || !yieldOp)
      return mlir::failure();

    auto loadOp = storeOp.getValue().getDefiningOp<mlir::memref::LoadOp>();
    if (!loadOp || loadOp.getMemref() != storeOp.getMemref())
      return mlir::failure();

    auto memrefType = llvm::dyn_cast<mlir::MemRefType>(loadOp.getMemref().getType());
    if (!memrefType || !memrefType.getElementType().isF32())
      return mlir::failure();

    auto loc = loop.getLoc();
    auto vecType = mlir::VectorType::get({vectorWidth}, rewriter.getF32Type());
    auto newStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, vectorWidth);

    auto vecLoop = rewriter.create<mlir::scf::ForOp>(loc, loop.getLowerBound(),
                                                     loop.getUpperBound(), newStep);
    vecLoop->setAttr("qfx.vectorized", rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    mlir::Value idx = vecLoop.getInductionVar();
    mlir::Value vec = rewriter.create<mlir::vector::TransferReadOp>(
        loc, vecType, loadOp.getMemref(), mlir::ValueRange{idx});
    rewriter.create<mlir::vector::TransferWriteOp>(loc, vec, storeOp.getMemref(),
                                                   mlir::ValueRange{idx});
    rewriter.eraseOp(loop);
    return mlir::success();
  }

  int64_t vectorWidth;
};

struct VectorizeLoopsPass
    : public mlir::PassWrapper<VectorizeLoopsPass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeLoopsPass)

  explicit VectorizeLoopsPass(int64_t width) : vectorWidth(width) {}

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeSimpleLoopPattern>(&getContext(), vectorWidth);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  llvm::StringRef getArgument() const final { return "qfx-vectorize"; }
  llvm::StringRef getDescription() const final {
    return "Vectorize simple elementwise loops (AVX2/AVX-512 width)";
  }

  int64_t vectorWidth = 8;
};

} // namespace

std::unique_ptr<mlir::Pass> createVectorizeLoopsPass(int64_t vectorWidth) {
  return std::make_unique<VectorizeLoopsPass>(vectorWidth);
}

} // namespace qfx
