#include <benchmark/benchmark.h>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>

// User-defined literals for byte sizes (requires C++14+)
constexpr std::size_t operator"" _B(unsigned long long k) { return k; }
constexpr std::size_t operator"" _KB(unsigned long long k) {
  return k * 1024ULL;
}
constexpr std::size_t operator"" _MB(unsigned long long m) {
  return m * 1024ULL * 1024ULL;
}
constexpr std::size_t operator"" _GB(unsigned long long g) {
  return g * 1024ULL * 1024ULL * 1024ULL;
}

// Helper: convert size to human-readable string
static std::string human_readable_size(std::size_t size) {
  if (size % 1_GB == 0 && size >= 1_GB)
    return std::to_string(size / 1_GB) + "GB";
  if (size % 1_MB == 0 && size >= 1_MB)
    return std::to_string(size / 1_MB) + "MB";
  if (size % 1_KB == 0 && size >= 1_KB)
    return std::to_string(size / 1_KB) + "KB";
  return std::to_string(size) + "B";
}

// Core memcpy workload (no state.range usage)
static void memcpy_work(benchmark::State &state, std::size_t size) {
  // Allocate aligned buffers
  uint64_t *src = (uint64_t *)_aligned_malloc(size, 256);
  uint64_t *dst = (uint64_t *)_aligned_malloc(size, 256);
  if (!src || !dst) {
    state.SkipWithError("_aligned_malloc failed");
    return;
  }
  std::memset(src, 0x0, size);

  for (auto _ : state) {
    benchmark::DoNotOptimize(dst);
    int64_t nb_iter = size / 8;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nb_iter; i++) {
      dst[i] = src[i];
    }
    benchmark::ClobberMemory();
  }

  _aligned_free(src);
  _aligned_free(dst);
  state.SetBytesProcessed(uint64_t(state.iterations()) * size);
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  // Register benchmarks for sizes 1B,2B,4B,... up to 5GB, without showing the
  // raw arg value
  for (std::size_t size = 1; size <= 5_GB; size *= 2) {
    std::string name =
        std::string("BM_memcpy/size:") + human_readable_size(size);
    benchmark::RegisterBenchmark(name.c_str(), [size](benchmark::State &st) {
      memcpy_work(st, size);
    })->UseRealTime();
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
