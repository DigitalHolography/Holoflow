#include <benchmark/benchmark.h>
#include <cstdlib>
#include <cstring>
#include <memory>

constexpr std::size_t SIZE = 8 * 1024 * 1024;
constexpr std::size_t CACHE_LINE_SIZE = 64;
constexpr std::size_t PAGE_SIZE = 4096;
constexpr std::size_t NUM_PAGES = SIZE / PAGE_SIZE;

__declspec(noinline) void *memcpy_std(void *dst, const void *src,
                                      std::size_t count) {
  return std::memcpy(dst, src, count);
}

__declspec(noinline) void *memcpy_naive(void *dst, const void *src,
                                        std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  for (int64_t i = 0; i < (int64_t)count; i++) {
    d[i] = s[i];
  }

  return dst;
}

__declspec(noinline) void *memcpy_wide(void *dst, const void *src,
                                       std::size_t count) {
  auto *d = static_cast<uint64_t *>(dst);
  const auto *s = static_cast<const uint64_t *>(src);
  const auto n_iter = count / sizeof(uint64_t);

  for (int64_t i = 0; i < (int64_t)n_iter; i++) {
    d[i] = s[i];
  }

  return dst;
}

__declspec(noinline) void *memcpy_wide128(void *dst, const void *src,
                                          std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  for (int64_t i = 0; i < (int64_t)count; i += 16) {
    _mm_stream_ps((float *)&d[i], _mm_load_ps((float *)&s[i]));
  }

  _mm_sfence();
  return dst;
}

static void BM_memcpy_std(benchmark::State &state) {
  auto src = std::make_unique<uint8_t[]>(SIZE);
  auto dst = std::make_unique<uint8_t[]>(SIZE);

  std::memset(src.get(), 0xA5, SIZE);

  for (auto _ : state) {
    memcpy_std(dst.get(), src.get(), SIZE);
    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * SIZE);
}

BENCHMARK(BM_memcpy_std);

static void BM_memcpy_naive(benchmark::State &state) {
  auto src = std::make_unique<uint8_t[]>(SIZE);
  auto dst = std::make_unique<uint8_t[]>(SIZE);

  std::memset(src.get(), 0xA5, SIZE);

  for (auto _ : state) {
    memcpy_naive(dst.get(), src.get(), SIZE);
    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * SIZE);
}

BENCHMARK(BM_memcpy_naive);

static void BM_memcpy_wide(benchmark::State &state) {
  auto src = std::make_unique<uint8_t[]>(SIZE);
  auto dst = std::make_unique<uint8_t[]>(SIZE);

  std::memset(src.get(), 0xA5, SIZE);

  for (auto _ : state) {
    memcpy_wide(dst.get(), src.get(), SIZE);
    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * SIZE);
}

BENCHMARK(BM_memcpy_wide);

static void BM_memcpy_wide128(benchmark::State &state) {
  auto src = std::make_unique<uint8_t[]>(SIZE);
  auto dst = std::make_unique<uint8_t[]>(SIZE);

  std::memset(src.get(), 0xA5, SIZE);

  for (auto _ : state) {
    memcpy_wide128(dst.get(), src.get(), SIZE);
    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * SIZE);
}

BENCHMARK(BM_memcpy_wide128);

BENCHMARK_MAIN();
