#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace holotask::syncs::detail {

void launch_resize_bilinear_cuda(const void *source, void *destination, std::size_t batch,
                                 std::size_t source_height, std::size_t source_width,
                                 std::size_t source_frame_stride, std::size_t source_row_stride,
                                 std::size_t destination_height, std::size_t destination_width,
                                 std::size_t element_size, cudaStream_t stream);

} // namespace holotask::syncs::detail
