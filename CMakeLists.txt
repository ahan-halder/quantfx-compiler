cmake_minimum_required(VERSION 3.28)
project(quantfx-compiler VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_program(LLVM_CONFIG NAMES llvm-config-18 llvm-config REQUIRED)
execute_process(
  COMMAND ${LLVM_CONFIG} --prefix
  OUTPUT_VARIABLE LLVM_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
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

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(compiler)
add_subdirectory(tools)
add_subdirectory(benchmarks)
add_subdirectory(tuner)
