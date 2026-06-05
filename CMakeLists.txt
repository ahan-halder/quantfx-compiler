cmake_minimum_required(VERSION 3.28)
project(quantfx-compiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer ccache when available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# LLVM / MLIR — use llvm-config-18 when present, else llvm-config
find_program(LLVM_CONFIG NAMES llvm-config-18 llvm-config REQUIRED)
execute_process(
  COMMAND ${LLVM_CONFIG} --cmakedir
  OUTPUT_VARIABLE LLVM_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_CMAKE_DIR}")
set(MLIR_DIR "${LLVM_CMAKE_DIR}/../mlir" CACHE PATH "MLIR CMake config directory")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# Subdirectories will be added as Phase 1 components land:
# add_subdirectory(compiler)
# add_subdirectory(runtime)
# add_subdirectory(benchmarks)
