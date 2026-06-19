get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(quantfx-compiler main.cpp Benchmark.cpp)

target_include_directories(quantfx-compiler PRIVATE
  ${PROJECT_SOURCE_DIR}/compiler/frontend
  ${PROJECT_SOURCE_DIR}/compiler/mlir/include
  ${PROJECT_BINARY_DIR}/compiler/mlir
  ${PROJECT_SOURCE_DIR}/compiler/backend/gpu
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  QFXConversion
  QFXTransforms
  QFXDialect
  qfxFrontend
  qfxGPU
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRParser
  MLIRSupport
)

set(LLVM_LINK_COMPONENTS
  Core
  Native
  NativeCodeGen
  Support
  Target
  AArch64CodeGen
  X86CodeGen
)

llvm_config(quantfx-compiler ${LLVM_LINK_COMPONENTS})
target_link_libraries(quantfx-compiler PRIVATE ${LIBS})

llvm_update_compile_flags(quantfx-compiler)
mlir_check_all_link_libraries(quantfx-compiler)
