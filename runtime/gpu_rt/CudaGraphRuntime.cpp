#include "CudaGraphRuntime.h"
#include <iostream>

#if QFX_HAS_CUDA
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            throw std::runtime_error("CUDA Error");                      \
        }                                                                \
    } while (0)
#endif

namespace qfx {
namespace rt {

CudaGraphRuntime::CudaGraphRuntime() : graph_(nullptr), graphExec_(nullptr), isCaptured_(false) {
#if QFX_HAS_CUDA
    CHECK_CUDA(cudaStreamCreate(&stream_));
#else
    stream_ = nullptr;
#endif
}

CudaGraphRuntime::~CudaGraphRuntime() {
#if QFX_HAS_CUDA
    if (graphExec_) cudaGraphExecDestroy(graphExec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (stream_) cudaStreamDestroy(stream_);
#endif
}

void CudaGraphRuntime::beginCapture() {
#if QFX_HAS_CUDA
    CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
#else
    throw std::runtime_error("CUDA SDK not installed: cannot capture CUDA graph.");
#endif
}

void CudaGraphRuntime::endCapture() {
#if QFX_HAS_CUDA
    CHECK_CUDA(cudaStreamEndCapture(stream_, &graph_));
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
    CHECK_CUDA(cudaGraphInstantiate(&graphExec_, graph_, 0));
#else
    CHECK_CUDA(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
#endif
    isCaptured_ = true;
#else
    throw std::runtime_error("CUDA SDK not installed: cannot instantiate CUDA graph.");
#endif
}

void CudaGraphRuntime::launch() {
    if (!isCaptured_) {
        throw std::runtime_error("Cannot launch CUDA graph: not captured yet.");
    }
#if QFX_HAS_CUDA
    CHECK_CUDA(cudaGraphLaunch(graphExec_, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));
#else
    throw std::runtime_error("CUDA SDK not installed: cannot launch CUDA graph.");
#endif
}

} // namespace rt
} // namespace qfx
