#include "resize_cuda.cuh"

#include <cstdint>

#include "curaii/cuda.hh"

namespace holotask::syncs::detail {
namespace {

template <typename T>
__global__ void resize_bilinear_kernel(const T *source, T *destination, std::size_t batch,
                                       std::size_t source_height, std::size_t source_width,
                                       std::size_t source_frame_stride, std::size_t source_row_stride,
                                       std::size_t destination_height,
                                       std::size_t destination_width) {
  const std::size_t output_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_size = batch * destination_height * destination_width;
  if (output_index >= output_size)
    return;

  const std::size_t frame = output_index / (destination_height * destination_width);
  const std::size_t output_offset = output_index % (destination_height * destination_width);
  const std::size_t y = output_offset / destination_width;
  const std::size_t x = output_offset % destination_width;

  const double fy = destination_height == 1
                        ? 0.0
                        : static_cast<double>(y) * (source_height - 1) /
                              static_cast<double>(destination_height - 1);
  const auto y0 = static_cast<std::size_t>(fy);
  const auto y1 = y0 + 1 < source_height ? y0 + 1 : source_height - 1;
  const double wy = fy - static_cast<double>(y0);

  const double fx = destination_width == 1
                        ? 0.0
                        : static_cast<double>(x) * (source_width - 1) /
                              static_cast<double>(destination_width - 1);
  const auto x0 = static_cast<std::size_t>(fx);
  const auto x1 = x0 + 1 < source_width ? x0 + 1 : source_width - 1;
  const double wx = fx - static_cast<double>(x0);

  const auto *frame_source = reinterpret_cast<const std::uint8_t *>(source) +
                              frame * source_frame_stride;
  const auto *row0 = reinterpret_cast<const T *>(frame_source + y0 * source_row_stride);
  const auto *row1 = reinterpret_cast<const T *>(frame_source + y1 * source_row_stride);
  const double top = static_cast<double>(row0[x0]) * (1.0 - wx) + row0[x1] * wx;
  const double bottom = static_cast<double>(row1[x0]) * (1.0 - wx) + row1[x1] * wx;
  destination[output_index] =
      static_cast<T>(floor(top * (1.0 - wy) + bottom * wy + 0.5));
}

} // namespace

void launch_resize_bilinear_cuda(const void *source, void *destination, std::size_t batch,
                                 std::size_t source_height, std::size_t source_width,
                                 std::size_t source_frame_stride, std::size_t source_row_stride,
                                 std::size_t destination_height, std::size_t destination_width,
                                 std::size_t element_size, cudaStream_t stream) {
  const auto output_size = batch * destination_height * destination_width;
  constexpr unsigned int block_size = 256;
  const auto grid_size = static_cast<unsigned int>((output_size + block_size - 1) / block_size);
  if (element_size == sizeof(std::uint8_t)) {
    resize_bilinear_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const std::uint8_t *>(source), static_cast<std::uint8_t *>(destination), batch,
        source_height, source_width, source_frame_stride, source_row_stride, destination_height,
        destination_width);
  } else {
    resize_bilinear_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const std::uint16_t *>(source), static_cast<std::uint16_t *>(destination), batch,
        source_height, source_width, source_frame_stride, source_row_stride, destination_height,
        destination_width);
  }
  CUDA_CHECK(cudaGetLastError());
}

} // namespace holotask::syncs::detail
