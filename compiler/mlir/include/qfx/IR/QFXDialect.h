#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "QFXDialect.h.inc"

namespace qfx {
class QFXDialect;
} // namespace qfx

#define GET_OP_CLASSES
#include "QFXOps.h.inc"
