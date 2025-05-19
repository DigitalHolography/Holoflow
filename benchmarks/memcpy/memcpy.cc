#include <benchmark/benchmark.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

constexpr std::size_t SIZE = 1024 * 1024 * 1024;
constexpr std::size_t CACHE_LINE_SIZE = 64;
constexpr std::size_t PAGE_SIZE = 4096;
constexpr std::size_t NUM_PAGES = SIZE / PAGE_SIZE;
constexpr std::size_t NUM_THREADS = 8;
constexpr std::size_t GRAIN = SIZE / NUM_THREADS;

inline void flush_cache(const void *p, std::size_t bytes) {
  const auto *c = static_cast<const uint8_t *>(p);
  for (std::size_t i = 0; i < bytes; i += CACHE_LINE_SIZE) {
    _mm_clflushopt(c + i);
  }

  _mm_mfence();
}

template <auto MemcpyFn> static void BM_memcpy(benchmark::State &state) {
  auto src = std::make_unique<uint8_t[]>(SIZE);
  auto dst = std::make_unique<uint8_t[]>(SIZE);
  std::memset(src.get(), 0xA5, SIZE);

  for (auto _ : state) {
    MemcpyFn(dst.get(), src.get(), SIZE);
    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();

    // state.PauseTiming();
    // flush_cache(src.get(), SIZE);
    // flush_cache(dst.get(), SIZE);
    // state.ResumeTiming();
  }
  state.SetBytesProcessed(state.iterations() * SIZE);
}

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

__declspec(noinline) void *memcpy_wide4(void *dst, const void *src,
                                        std::size_t count) {
  auto *d = static_cast<uint64_t *>(dst);
  const auto *s = static_cast<const uint64_t *>(src);
  const auto n_iter = count / sizeof(uint64_t);

  for (int64_t i = 0; i < (int64_t)n_iter; i++) {
    d[i] = s[i];
  }

  return dst;
}

__declspec(noinline) void *memcpy_wide16(void *dst, const void *src,
                                         std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  for (int64_t i = 0; i < (int64_t)count; i += 16) {
    _mm_stream_ps((float *)&d[i], _mm_load_ps((float *)&s[i]));
  }

  _mm_sfence();
  return dst;
}

__declspec(noinline) void *memcpy_wide64(void *dst, const void *src,
                                         std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  for (int64_t i = 0; i < (int64_t)count; i += 16 * 4) {
    _mm_stream_ps((float *)&d[i + 16 * 0],
                  _mm_load_ps((float *)&s[i + 16 * 0]));
    _mm_stream_ps((float *)&d[i + 16 * 1],
                  _mm_load_ps((float *)&s[i + 16 * 1]));
    _mm_stream_ps((float *)&d[i + 16 * 2],
                  _mm_load_ps((float *)&s[i + 16 * 2]));
    _mm_stream_ps((float *)&d[i + 16 * 3],
                  _mm_load_ps((float *)&s[i + 16 * 3]));
  }

  _mm_sfence();
  return dst;
}

__declspec(noinline) void *memcpy_wide128(void *dst, const void *src,
                                          std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  for (int64_t i = 0; i < (int64_t)count; i += 16 * 8) {
    _mm_stream_ps((float *)&d[i + 16 * 0],
                  _mm_load_ps((float *)&s[i + 16 * 0]));
    _mm_stream_ps((float *)&d[i + 16 * 1],
                  _mm_load_ps((float *)&s[i + 16 * 1]));
    _mm_stream_ps((float *)&d[i + 16 * 2],
                  _mm_load_ps((float *)&s[i + 16 * 2]));
    _mm_stream_ps((float *)&d[i + 16 * 3],
                  _mm_load_ps((float *)&s[i + 16 * 3]));
    _mm_stream_ps((float *)&d[i + 16 * 4],
                  _mm_load_ps((float *)&s[i + 16 * 4]));
    _mm_stream_ps((float *)&d[i + 16 * 5],
                  _mm_load_ps((float *)&s[i + 16 * 5]));
    _mm_stream_ps((float *)&d[i + 16 * 6],
                  _mm_load_ps((float *)&s[i + 16 * 6]));
    _mm_stream_ps((float *)&d[i + 16 * 7],
                  _mm_load_ps((float *)&s[i + 16 * 7]));
  }

  _mm_sfence();
  return dst;
}

__declspec(noinline) void *memcpy_std_mt(void *dst, const void *src,
                                         std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, GRAIN),
                    [&](const tbb::blocked_range<std::size_t> &r) {
                      std::memcpy(d + r.begin(), s + r.begin(), r.size());
                    });
  return dst;
}

__declspec(noinline) void *memcpy_naive_mt(void *dst, const void *src,
                                           std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, GRAIN),
                    [&](const tbb::blocked_range<std::size_t> &r) {
                      for (std::size_t i = r.begin(); i < r.end(); ++i) {
                        d[i] = s[i];
                      }
                    });
  return dst;
}

__declspec(noinline) void *memcpy_wide4_mt(void *dst, const void *src,
                                           std::size_t count) {
  auto *d = static_cast<uint64_t *>(dst);
  const auto *s = static_cast<const uint64_t *>(src);
  const auto n_iter = count / sizeof(uint64_t);

#pragma omp parallel for num_threads(8) schedule(static)
  for (int64_t i = 0; i < (int64_t)n_iter; i++) {
    d[i] = s[i];
  }

  return dst;
}

__declspec(noinline) void *memcpy_wide16_mt(void *dst, const void *src,
                                            std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, GRAIN),
                    [&](tbb::blocked_range<std::size_t> r) {
                      for (std::size_t i = r.begin(); i < r.end(); i += 16) {
                        _mm_stream_ps((float *)&d[i],
                                      _mm_load_ps((float *)&s[i]));
                      }
                    });

  _mm_sfence();
  return dst;
}

__declspec(noinline) void *memcpy_wide64_mt(void *dst, const void *src,
                                            std::size_t count) {
  auto *d = static_cast<uint8_t *>(dst);
  const auto *s = static_cast<const uint8_t *>(src);

  for (int64_t i = 0; i < (int64_t)count; i += 16 * 4) {
    _mm_stream_ps((float *)&d[i + 16 * 0],
                  _mm_load_ps((float *)&s[i + 16 * 0]));
    _mm_stream_ps((float *)&d[i + 16 * 1],
                  _mm_load_ps((float *)&s[i + 16 * 1]));
    _mm_stream_ps((float *)&d[i + 16 * 2],
                  _mm_load_ps((float *)&s[i + 16 * 2]));
    _mm_stream_ps((float *)&d[i + 16 * 3],
                  _mm_load_ps((float *)&s[i + 16 * 3]));
  }

  _mm_sfence();
  return dst;
}

BENCHMARK_TEMPLATE(BM_memcpy, memcpy_std)->Name("std")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_naive)->Name("naive")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_wide4)->Name("4B")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_wide16)->Name("16B")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_std_mt)->Name("std_mt")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_naive_mt)->Name("naive_mt")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_wide4_mt)->Name("4B_mt")->MinTime(5.0);
BENCHMARK_TEMPLATE(BM_memcpy, memcpy_wide16_mt)->Name("16B_mt")->MinTime(5.0);

BENCHMARK_MAIN();
