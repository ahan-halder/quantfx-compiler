#pragma once

#include <cuda_runtime.h>
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
