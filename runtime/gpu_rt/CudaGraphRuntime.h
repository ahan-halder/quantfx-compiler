#pragma once

#if defined(__has_include) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define QFX_HAS_CUDA 1
#else
// Fallback definitions when CUDA SDK is not installed on the system
using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
#define QFX_HAS_CUDA 0
#endif

#include <stdexcept>
#include <functional>

namespace qfx {
namespace rt {

class CudaGraphRuntime {
public:
    CudaGraphRuntime();
    ~CudaGraphRuntime();

    // Start capturing operations on the stream
    void beginCapture();

    // End capturing and instantiate the graph
    void endCapture();

    // Execute the captured graph
    void launch();

    // Get the stream for capturing
    cudaStream_t getStream() const { return stream_; }

private:
    cudaStream_t stream_;
    cudaGraph_t graph_;
    cudaGraphExec_t graphExec_;
    bool isCaptured_;
};

} // namespace rt
} // namespace qfx
