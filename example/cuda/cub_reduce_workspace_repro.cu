#include <cstdio>
#include <cstdlib>
#include <cub/device/device_reduce.cuh>

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

int main() {
  const int    count      = 1024 * 1024;
  const size_t wide_count = count;
  float       *input = nullptr, *output = nullptr;
  check(cudaMalloc(&input, count * sizeof(float)));
  check(cudaMalloc(&output, sizeof(float)));
  check(cudaMemset(input, 0, count * sizeof(float)));

  size_t int_bytes = 0, size_t_bytes = 0;
  check(cub::DeviceReduce::Min(nullptr, int_bytes, input, output, count));
  check(cub::DeviceReduce::Min(nullptr, size_t_bytes, input, output, wide_count));
  std::printf("int count:    %zu bytes\n", int_bytes);
  std::printf("size_t count: %zu bytes\n", size_t_bytes);

  void *scratch = nullptr;
  check(cudaMalloc(&scratch, size_t_bytes));
  auto status = cub::DeviceReduce::Min(scratch, size_t_bytes, input, output, count);
  std::printf("size_t query / int execution: %s\n", cudaGetErrorString(status));

  check(cudaFree(scratch));
  check(cudaFree(output));
  check(cudaFree(input));
}
