#include "CudaGraphRuntime.h"
#include <iostream>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            throw std::runtime_error("CUDA Error");                      \
        }                                                                \
    } while (0)

namespace qfx {
namespace rt {

CudaGraphRuntime::CudaGraphRuntime() : graph_(nullptr), graphExec_(nullptr), isCaptured_(false) {
    CHECK_CUDA(cudaStreamCreate(&stream_));
}

CudaGraphRuntime::~CudaGraphRuntime() {
    if (graphExec_) cudaGraphExecDestroy(graphExec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (stream_) cudaStreamDestroy(stream_);
}

void CudaGraphRuntime::beginCapture() {
    CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
}

void CudaGraphRuntime::endCapture() {
    CHECK_CUDA(cudaStreamEndCapture(stream_, &graph_));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
    isCaptured_ = true;
}

void CudaGraphRuntime::launch() {
    if (!isCaptured_) {
        throw std::runtime_error("Cannot launch CUDA graph: not captured yet.");
    }
    CHECK_CUDA(cudaGraphLaunch(graphExec_, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

} // namespace rt
} // namespace qfx
