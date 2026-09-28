#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "bug.hh"
#include "holoflow/core/tensor.hh"

namespace holotask::sinks::detail {

struct RecordingGeometry {
  std::uint8_t  bits_per_pixel;
  std::uint32_t frame_width;
  std::uint32_t frame_height;
};

inline std::uint8_t bits_per_pixel_for(holoflow::core::DType dtype) {
  switch (dtype) {
  case holoflow::core::DType::U8:
    return 8;
  case holoflow::core::DType::U16:
    return 16;
  default:
    HOLOVIBES_BUG("Unsupported recording dtype: {}", static_cast<int>(dtype));
  }
}

inline RecordingGeometry recording_geometry_from_desc(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != 3)
    throw std::invalid_argument("recording input must have rank 3");
  return {.bits_per_pixel = bits_per_pixel_for(desc.dtype),
          .frame_width    = static_cast<std::uint32_t>(desc.shape[2]),
          .frame_height   = static_cast<std::uint32_t>(desc.shape[1])};
}

inline std::size_t recording_frame_byte_size(const RecordingGeometry &geometry) {
  return static_cast<std::size_t>(geometry.frame_width) * geometry.frame_height *
         geometry.bits_per_pixel / 8;
}

inline bool same_geometry(const RecordingGeometry &a, const RecordingGeometry &b) {
  return a.bits_per_pixel == b.bits_per_pixel && a.frame_width == b.frame_width &&
         a.frame_height == b.frame_height;
}

} // namespace holotask::sinks::detail
