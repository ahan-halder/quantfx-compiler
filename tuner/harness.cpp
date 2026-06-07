// Compile-and-benchmark harness: invokes quantfx-compiler --benchmark with a JSON config.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string readFile(const std::string &path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool extractJsonLine(const std::string &output, std::string &jsonLine) {
  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.front() == '{') {
      jsonLine = line;
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  std::string kernel;
  std::string compiler = "quantfx-compiler";
  std::string configPath;
  int warmup = 100;
  int iters = 1000;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--kernel" && i + 1 < argc)
      kernel = argv[++i];
    else if (arg == "--compiler" && i + 1 < argc)
      compiler = argv[++i];
    else if (arg == "--config" && i + 1 < argc)
      configPath = argv[++i];
    else if (arg == "--warmup" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (arg == "--help") {
      std::cout << "Usage: qfx-harness --kernel file.qfx [--compiler path] "
                   "[--config cfg.json] [--warmup N] [--iters N]\n";
      return 0;
    }
  }

  if (kernel.empty()) {
    std::cerr << "missing --kernel\n";
    return 1;
  }

  std::ostringstream cmd;
  cmd << compiler << " " << kernel << " --benchmark"
      << " --bench-warmup=" << warmup << " --bench-iters=" << iters;

  if (!configPath.empty()) {
    std::string cfg = readFile(configPath);
    if (cfg.find("\"opt_level\"") != std::string::npos) {
      if (cfg.find("\"streaming\": true") != std::string::npos)
        cmd << " --streaming";
      if (cfg.find("\"window_fusion\": false") != std::string::npos)
        cmd << " --no-fusion";
      if (cfg.find("\"loop_tiling\": false") != std::string::npos)
        cmd << " --no-tiling";
      if (cfg.find("\"prefetch\": false") != std::string::npos)
        cmd << " --no-prefetch";
      if (cfg.find("\"vectorize\": true") != std::string::npos)
        cmd << " --vectorize";
    }
  }

  auto start = std::chrono::steady_clock::now();
  std::string fullCmd = cmd.str() + " 2>&1";
  FILE *pipe = popen(fullCmd.c_str(), "r");
  if (!pipe) {
    std::cerr << "failed to launch compiler\n";
    return 1;
  }

  char buffer[4096];
  std::string output;
  while (fgets(buffer, sizeof(buffer), pipe))
    output += buffer;
  int rc = pclose(pipe);
  auto end = std::chrono::steady_clock::now();
  double harnessMs =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

  std::string jsonLine;
  if (!extractJsonLine(output, jsonLine)) {
    std::cerr << output;
    return rc != 0 ? rc : 1;
  }

  std::cout << jsonLine << "\n";
  std::cerr << "harness_overhead_ms=" << harnessMs << "\n";
  return rc == 0 ? 0 : 1;
}
