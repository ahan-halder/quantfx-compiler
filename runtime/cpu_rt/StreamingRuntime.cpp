#include "StreamingRuntime.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace qfx {
namespace rt {

// StreamingPipeline: feeds tick data from a ring buffer into a kernel function
// at maximum throughput, while recording per-tick latency.

class StreamingPipeline {
public:
    using KernelFn = void (*)(const float* prices, const float* volumes,
                              float* output, int n, int window);

    StreamingPipeline(int window_size, int core_id = 0)
        : window_size_(window_size), core_id_(core_id),
          input_ring_(4096), running_(false) {}

    void start(KernelFn kernel) {
        lock_memory();
        pin_thread_to_core(core_id_);
        running_.store(true, std::memory_order_release);
        kernel_ = kernel;
    }

    void stop() {
        running_.store(false, std::memory_order_release);
    }

    // Feed a single tick into the ring buffer (producer side)
    bool feed(float price, float volume) {
        struct Tick { float price; float volume; };
        // Pack price/volume into a single float-pair via ring buffer
        // For simplicity, use two separate pushes
        if (!input_ring_.push(price)) return false;
        if (!input_ring_.push(volume)) return false;
        return true;
    }

    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

private:
    int window_size_;
    int core_id_;
    RingBuffer<float> input_ring_;
    std::atomic<bool> running_;
    KernelFn kernel_ = nullptr;
};

} // namespace rt
} // namespace qfx
