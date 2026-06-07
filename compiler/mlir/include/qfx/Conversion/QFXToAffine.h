#pragma once

#include "mlir/Pass/Pass.h"

namespace qfx {

std::unique_ptr<mlir::Pass> createQFXToAffinePass();

} // namespace qfx
