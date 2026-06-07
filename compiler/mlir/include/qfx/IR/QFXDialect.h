#pragma once

#include "mlir/IR/Dialect.h"

#include "qfx/IR/QFXDialect.h.inc"

namespace qfx {
class QFXDialect;
} // namespace qfx

#define GET_OP_CLASSES
#include "qfx/IR/QFXOps.h.inc"
