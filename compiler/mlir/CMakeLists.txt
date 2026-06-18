include(TableGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/qfx/IR/QFXOps.td)
list(APPEND MLIR_TABLEGEN_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/include")

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

add_mlir_library(QFXTransforms
  lib/Transforms/WindowFusion.cpp
  lib/Transforms/SlidingWindowUpdate.cpp
  lib/Transforms/LoopTiling.cpp
  lib/Transforms/VectorizeLoops.cpp
  lib/Transforms/PrefetchInsertion.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  QFXDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
)

add_mlir_library(QFXConversion
  lib/Conversion/HirToMLIR.cpp
  lib/Conversion/QFXToAffine.cpp
  lib/Conversion/LowerToLLVM.cpp
  lib/Conversion/PassPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  QFXDialect
  QFXTransforms
  qfxFrontend
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRVectorToLLVM
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

target_include_directories(QFXTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
