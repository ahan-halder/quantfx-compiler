#pragma once

#include "mlir/IR/Dialect.h"

#include "QFXDialect.h.inc"

namespace qfx {
class QFXDialect;
} // namespace qfx

#define GET_OP_CLASSES
#include "QFXOps.h.inc"
