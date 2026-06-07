include(TableGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/qfx/IR/QFXOps.td)

mlir_tablegen(QFXOps.h.inc -gen-op-decls)
mlir_tablegen(QFXOps.cpp.inc -gen-op-defs)
mlir_tablegen(QFXDialect.h.inc -gen-dialect-decls -dialect=qfx)
mlir_tablegen(QFXDialect.cpp.inc -gen-dialect-defs -dialect=qfx)
add_public_tablegen_target(MLIRQFXIncGen)

add_mlir_dialect_library(QFXDialect
  lib/Dialect/QFXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  DEPENDS
  MLIRQFXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

add_mlir_library(QFXConversion
  lib/Conversion/HirToMLIR.cpp
  lib/Conversion/QFXToAffine.cpp
  lib/Conversion/LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  QFXDialect
  qfxFrontend
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRIndexToLLVM
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRPass
  MLIRTransforms
)

target_include_directories(QFXConversion PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/compiler/frontend
)
