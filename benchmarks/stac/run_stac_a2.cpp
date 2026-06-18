#include "../tick/TickDataReplay.h"
#include "../../runtime/cpu_rt/StreamingRuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace qfx::rt;
using namespace qfx::tick;

// ---------------------------------------------------------------------------
// Streaming analytics kernels (mimic compiled .qfx output)
// ---------------------------------------------------------------------------

struct StreamingVwap {
    float prices[256];   // circular buffer, max window 256
    float volumes[256];
    int head = 0;
    int count = 0;
    int window;

    explicit StreamingVwap(int w) : window(w) {
        std::memset(prices, 0, sizeof(prices));
        std::memset(volumes, 0, sizeof(volumes));
    }

    float update(float price, float volume) {
        prices[head % window] = price;
        volumes[head % window] = volume;
        head++;
        count = std::min(count + 1, window);

        float sum_pv = 0.0f, sum_v = 0.0f;
        for (int i = 0; i < count; ++i) {
            sum_pv += prices[i] * volumes[i];
            sum_v += volumes[i];
        }
        return (sum_v > 1e-8f) ? (sum_pv / sum_v) : 0.0f;
    }
};

struct StreamingStats {
    float buf[256];
    int head = 0;
    int count = 0;
    int window;

    explicit StreamingStats(int w) : window(w) {
        std::memset(buf, 0, sizeof(buf));
    }

    void push(float x) {
        buf[head % window] = x;
        head++;
        count = std::min(count + 1, window);
    }

    float mean() const {
        float s = 0.0f;
        for (int i = 0; i < count; ++i) s += buf[i];
        return s / static_cast<float>(count);
    }

    float stddev() const {
        float m = mean();
        float var = 0.0f;
        for (int i = 0; i < count; ++i) {
            float d = buf[i] - m;
            var += d * d;
        }
        return std::sqrt(var / static_cast<float>(count));
    }
};

struct StreamingEma {
    float value = 0.0f;
    float alpha;
    bool initialized = false;

    explicit StreamingEma(float a) : alpha(a) {}

    float update(float x) {
        if (!initialized) { value = x; initialized = true; return value; }
        value = alpha * x + (1.0f - alpha) * value;
        return value;
    }
};

// ---------------------------------------------------------------------------
// Latency statistics
// ---------------------------------------------------------------------------

struct LatencyStats {
    long long min_ns;
    long long max_ns;
    long long median_ns;
    long long p99_ns;
    long long p999_ns;
    double mean_ns;
    double stddev_ns;
    size_t count;
};

LatencyStats compute_stats(std::vector<long long>& samples) {
    LatencyStats s{};
    s.count = samples.size();
    if (samples.empty()) return s;

    std::sort(samples.begin(), samples.end());
    s.min_ns = samples.front();
    s.max_ns = samples.back();
    s.median_ns = samples[samples.size() / 2];
    s.p99_ns = samples[static_cast<size_t>(samples.size() * 0.99)];
    s.p999_ns = samples[static_cast<size_t>(samples.size() * 0.999)];

    double sum = 0.0;
    for (auto v : samples) sum += static_cast<double>(v);
    s.mean_ns = sum / static_cast<double>(s.count);

    double var = 0.0;
    for (auto v : samples) {
        double d = static_cast<double>(v) - s.mean_ns;
        var += d * d;
    }
    s.stddev_ns = std::sqrt(var / static_cast<double>(s.count));

    return s;
}

void print_stats(const char* label, const LatencyStats& s) {
    std::printf("  %-24s  %6zu samples | median %5lld ns | p99 %5lld ns | p99.9 %5lld ns | "
                "mean %.1f ns | std %.1f ns | min %lld ns | max %lld ns\n",
                label, s.count, s.median_ns, s.p99_ns, s.p999_ns,
                s.mean_ns, s.stddev_ns, s.min_ns, s.max_ns);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string dataPath;
    int windowSize = 20;
    bool jsonOutput = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) dataPath = argv[++i];
        else if (arg == "--window" && i + 1 < argc) windowSize = std::atoi(argv[++i]);
        else if (arg == "--json") jsonOutput = true;
    }

    if (dataPath.empty()) {
        std::cerr << "Usage: " << argv[0] << " --data <lobster_sample.csv> [--window 20] [--json]\n";
        return 1;
    }

    TickDataReplay replay(dataPath);
    if (!replay.load()) return 1;

    const auto& messages = replay.getMessages();

    // Setup low-latency environment
    lock_memory();
    pin_thread_to_core(0);

    std::printf("quantfx-compiler STAC-A2 Benchmark\n");
    std::printf("===================================\n");
    std::printf("Tick messages loaded: %zu\n", messages.size());
    std::printf("Window size:         %d\n", windowSize);
    std::printf("CPU pinning:         core 0\n");
    std::printf("Memory locking:      mlockall(MCL_CURRENT|MCL_FUTURE)\n\n");

    // --- Benchmark 1: Streaming VWAP ---
    {
        StreamingVwap vwap(windowSize);
        std::vector<long long> lats;
        lats.reserve(messages.size());
        for (const auto& msg : messages) {
            if (msg.type != 4 && msg.type != 5) continue;
            auto t0 = std::chrono::steady_clock::now();
            volatile float r = vwap.update(msg.price, static_cast<float>(msg.size));
            (void)r;
            auto t1 = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        auto s = compute_stats(lats);
        print_stats("Streaming VWAP", s);
    }

    // --- Benchmark 2: Streaming rolling mean + std (fused) ---
    {
        StreamingStats stats(windowSize);
        std::vector<long long> lats;
        lats.reserve(messages.size());
        for (const auto& msg : messages) {
            if (msg.type != 4 && msg.type != 5) continue;
            auto t0 = std::chrono::steady_clock::now();
            stats.push(msg.price);
            volatile float m = stats.mean();
            volatile float sd = stats.stddev();
            (void)m; (void)sd;
            auto t1 = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        auto s = compute_stats(lats);
        print_stats("Fused Mean+Std", s);
    }

    // --- Benchmark 3: Z-Score signal (VWAP + std + zscore) ---
    {
        StreamingVwap vwap(windowSize);
        StreamingStats stats(windowSize);
        std::vector<long long> lats;
        lats.reserve(messages.size());
        for (const auto& msg : messages) {
            if (msg.type != 4 && msg.type != 5) continue;
            auto t0 = std::chrono::steady_clock::now();
            float v = vwap.update(msg.price, static_cast<float>(msg.size));
            stats.push(msg.price);
            float sd = stats.stddev();
            volatile float z = (sd > 1e-8f) ? (msg.price - v) / sd : 0.0f;
            (void)z;
            auto t1 = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        auto s = compute_stats(lats);
        print_stats("Z-Score Signal", s);
    }

    // --- Benchmark 4: EMA Crossover ---
    {
        StreamingEma fast(0.1f), slow(0.05f);
        std::vector<long long> lats;
        lats.reserve(messages.size());
        float prevFast = 0.0f, prevSlow = 0.0f;
        for (const auto& msg : messages) {
            if (msg.type != 4 && msg.type != 5) continue;
            auto t0 = std::chrono::steady_clock::now();
            float f = fast.update(msg.price);
            float s = slow.update(msg.price);
            volatile float sig = (f > s && prevFast <= prevSlow) ? 1.0f : 0.0f;
            (void)sig;
            prevFast = f; prevSlow = s;
            auto t1 = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        auto s = compute_stats(lats);
        print_stats("EMA Crossover", s);
    }

    // --- Benchmark 5: Full analytics chain ---
    {
        StreamingVwap vwap(windowSize);
        StreamingStats stats(windowSize);
        StreamingEma fast(0.1f), slow(0.05f);
        std::vector<long long> lats;
        lats.reserve(messages.size());
        float prevFast = 0.0f, prevSlow = 0.0f;
        for (const auto& msg : messages) {
            if (msg.type != 4 && msg.type != 5) continue;
            auto t0 = std::chrono::steady_clock::now();

            float v = vwap.update(msg.price, static_cast<float>(msg.size));
            stats.push(msg.price);
            float m = stats.mean();
            float sd = stats.stddev();
            volatile float z = (sd > 1e-8f) ? (msg.price - v) / sd : 0.0f;
            float f = fast.update(msg.price);
            float s = slow.update(msg.price);
            volatile float cross = (f > s && prevFast <= prevSlow) ? 1.0f : 0.0f;
            (void)z; (void)cross; (void)m;
            prevFast = f; prevSlow = s;

            auto t1 = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        auto s = compute_stats(lats);
        print_stats("Full Chain (5 ops)", s);
    }

    std::printf("\n");
    return 0;
}
