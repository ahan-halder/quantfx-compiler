#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

float bench_rolling_mean(const float *input, float *output, int n, int window, int iters) {
  for (int iter = 0; iter < iters; ++iter) {
    for (int k = 0; k < n; ++k) {
      float sum = 0.f;
      for (int i = 0; i < window; ++i) {
        int j = k - i;
        if (j >= 0 && j < n)
          sum += input[j] / static_cast<float>(window);
      }
      output[k] = sum;
    }
  }
  return output[n / 2];
}

} // namespace

int main(int argc, char **argv) {
  int n = 10000;
  int window = 20;
  int iters = 10000;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc)
      n = std::atoi(argv[++i]);
    else if (arg == "--window" && i + 1 < argc)
      window = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
  }

  std::vector<float> input(n), output(n);
  for (int i = 0; i < n; ++i)
    input[i] = static_cast<float>(i % 97) * 0.01f - 0.5f;

  auto start = std::chrono::steady_clock::now();
  volatile float sink = bench_rolling_mean(input.data(), output.data(), n, window, iters);
  auto end = std::chrono::steady_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double perIter = static_cast<double>(us) / iters;
  std::printf("bench_rolling_mean n=%d window=%d iters=%d\n", n, window, iters);
  std::printf("  total: %.2f ms  per-iter: %.2f us  sink=%f\n", us / 1000.0, perIter, sink);
  return 0;
}
