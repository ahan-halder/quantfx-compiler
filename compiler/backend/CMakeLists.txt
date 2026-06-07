add_library(qfxGPU STATIC gpu/GPUCodeGen.cpp)

target_include_directories(qfxGPU PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu
)

target_link_libraries(qfxGPU PUBLIC
  qfxFrontend
  MLIRSupport
)

target_compile_features(qfxGPU PUBLIC cxx_std_20)
