add_library(qfx_cpu_rt STATIC cpu_rt/StreamingRuntime.cpp)
target_include_directories(qfx_cpu_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(qfx_cpu_rt PRIVATE cxx_std_20)
target_link_libraries(qfx_cpu_rt PUBLIC pthread)

add_library(qfx_gpu_rt STATIC gpu_rt/CudaGraphRuntime.cpp)
target_include_directories(qfx_gpu_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(qfx_gpu_rt PRIVATE cxx_std_20)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(qfx_gpu_rt PUBLIC CUDA::cudart)
endif()
