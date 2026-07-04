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
  QFXConversion
  QFXTransforms
  QFXDialect
  qfxFrontend
  qfxGPU
  ${dialect_libs}
  ${conversion_libs}
  MLIRToLLVMIRTranslationRegistration
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRParser
  MLIRSupport
)

target_link_libraries(quantfx-compiler PRIVATE ${LIBS} LLVM)

llvm_update_compile_flags(quantfx-compiler)
mlir_check_all_link_libraries(quantfx-compiler)
