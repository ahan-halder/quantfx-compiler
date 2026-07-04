#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
void test(mlir::OpBuilder &b) {
  b.create<mlir::arith::ConstantFloatOp>(b.getUnknownLoc(), llvm::APFloat(0.0f), b.getF32Type());
}
