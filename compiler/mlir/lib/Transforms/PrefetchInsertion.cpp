#include "qfx/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace qfx {
namespace {

struct PrefetchInsertionPass
    : public mlir::PassWrapper<PrefetchInsertionPass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchInsertionPass)

  explicit PrefetchInsertionPass(int64_t dist) : distance(dist) {}

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    getOperation().walk([&](mlir::memref::LoadOp load) {
      auto loop = load->getParentOfType<mlir::scf::ForOp>();
      if (!loop)
        return;

      auto idx = load.getIndices().front();
      auto idxConst = idx.getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!idxConst)
        return;

      builder.setInsertionPoint(load);
      auto loc = load.getLoc();
      auto distCst = builder.create<mlir::arith::ConstantIndexOp>(loc, distance);
      mlir::Value prefetchIdx = builder.create<mlir::arith::AddIOp>(loc, idx, distCst);
      builder.create<mlir::memref::PrefetchOp>(loc, load.getMemref(),
                                               mlir::ValueRange{prefetchIdx},
                                               /*isWrite=*/false, /*locality=*/3,
                                               /*isData=*/true);
    });
  }

  llvm::StringRef getArgument() const final { return "qfx-prefetch"; }
  llvm::StringRef getDescription() const final {
    return "Insert stride-based memref prefetches ahead of loads";
  }

  int64_t distance = 8;
};

} // namespace

std::unique_ptr<mlir::Pass> createPrefetchInsertionPass(int64_t distance) {
  return std::make_unique<PrefetchInsertionPass>(distance);
}

} // namespace qfx
