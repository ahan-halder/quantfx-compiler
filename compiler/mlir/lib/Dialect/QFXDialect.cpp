#include "qfx/IR/QFXDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "qfx/IR/QFXDialect.cpp.inc"

#define GET_OP_CLASSES
#include "qfx/IR/QFXOps.cpp.inc"

namespace qfx {

void QFXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "qfx/IR/QFXOps.cpp.inc"
      >();
}

} // namespace qfx
