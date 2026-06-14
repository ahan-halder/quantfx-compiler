#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <pthread.h>
#include <sys/mman.h>

namespace qfx {
namespace rt {

// Lock all current and future memory into RAM to prevent page faults (jitter)
inline void lock_memory() {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    std::cerr << "Warning: mlockall failed, jitter may occur (try running with higher privileges or setting ulimit -l unlimited).\n";
  }
}

// Pin current thread to a specific CPU core
inline void pin_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  pthread_t current_thread = pthread_self();
  int s = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    std::cerr << "Warning: pthread_setaffinity_np failed, jitter may occur.\n";
  }
}

template <typename T>
class RingBuffer {
public:
  RingBuffer(size_t capacity) : capacity_(capacity), mask_(capacity - 1) {
    // Capacity must be a power of 2 for fast masking
    if ((capacity & (capacity - 1)) != 0) {
      throw std::invalid_argument("RingBuffer capacity must be a power of 2");
    }
    buffer_.resize(capacity);
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
  }

  bool push(const T& item) {
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_acquire);

    if (head - tail >= capacity_) {
      return false; // Buffer full
    }

    buffer_[head & mask_] = item;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  bool pop(T& item) {
    size_t tail = tail_.load(std::memory_order_relaxed);
    size_t head = head_.load(std::memory_order_acquire);

    if (tail == head) {
      return false; // Buffer empty
    }

    item = buffer_[tail & mask_];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  size_t size() const {
    size_t head = head_.load(std::memory_order_acquire);
    size_t tail = tail_.load(std::memory_order_acquire);
    return head - tail;
  }

private:
  size_t capacity_;
  size_t mask_;
  std::vector<T> buffer_;
  alignas(64) std::atomic<size_t> head_;
  alignas(64) std::atomic<size_t> tail_;
};

} // namespace rt
} // namespace qfx
