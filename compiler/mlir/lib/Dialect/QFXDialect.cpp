#include "qfx/IR/QFXDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "QFXDialect.cpp.inc"

#define GET_OP_CLASSES
#include "QFXOps.cpp.inc"

namespace qfx {

void QFXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "QFXOps.cpp.inc"
      >();
}

} // namespace qfx
