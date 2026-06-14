#include "../tick/TickDataReplay.h"
#include "../../runtime/cpu_rt/StreamingRuntime.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace qfx::rt;
using namespace qfx::tick;

int main(int argc, char** argv) {
    std::string dataPath = "";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            dataPath = argv[++i];
        }
    }

    if (dataPath.empty()) {
        std::cerr << "Usage: " << argv[0] << " --data <lobster_sample.csv>\n";
        return 1;
    }

    TickDataReplay replay(dataPath);
    if (!replay.load()) {
        return 1;
    }

    const auto& messages = replay.getMessages();
    std::cout << "Loaded " << messages.size() << " tick messages.\n";

    // Setup low-latency environment
    lock_memory();
    pin_thread_to_core(0);

    // Simulate streaming execution (STAC-A2 workload: VWAP)
    int window_size = 20;
    std::vector<float> prices(window_size, 0.0f);
    std::vector<float> volumes(window_size, 0.0f);
    
    std::vector<long long> latencies_ns;
    latencies_ns.reserve(messages.size());

    int idx = 0;
    for (const auto& msg : messages) {
        if (msg.type != 4 && msg.type != 5) continue; // Only process executions

        auto start = std::chrono::steady_clock::now();

        prices[idx % window_size] = msg.price;
        volumes[idx % window_size] = static_cast<float>(msg.size);
        idx++;

        float sum_pv = 0.0f;
        float sum_v = 0.0f;
        for (int i = 0; i < window_size; ++i) {
            sum_pv += prices[i] * volumes[i];
            sum_v += volumes[i];
        }
        volatile float vwap = (sum_v > 0) ? (sum_pv / sum_v) : 0.0f;
        (void)vwap; // prevent optimization

        auto end = std::chrono::steady_clock::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        latencies_ns.push_back(ns);
    }

    if (latencies_ns.empty()) {
        std::cout << "No execution messages found to process.\n";
        return 0;
    }

    std::sort(latencies_ns.begin(), latencies_ns.end());
    long long median = latencies_ns[latencies_ns.size() / 2];
    long long p99 = latencies_ns[latencies_ns.size() * 99 / 100];

    std::cout << "Processed " << latencies_ns.size() << " executions.\n";
    std::cout << "Median Latency: " << median << " ns\n";
    std::cout << "p99 Latency:    " << p99 << " ns\n";

    return 0;
}
