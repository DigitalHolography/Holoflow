// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holofile/holofile.hh"

#include <system_error>

#include "logger.hh"

namespace holofile {

Exception::Exception(const std::string &message) : std::runtime_error(message) {}

Exception::~Exception() noexcept = default;

EndOfFileException::EndOfFileException() : Exception("Holofile: End of file reached") {}

IncompleteHeaderException::IncompleteHeaderException() : Exception("Holofile: Incomplete header") {}

InvalidMagicNumberException::InvalidMagicNumberException()
    : Exception("Holofile: Invalid magic number") {}

InvalidVersionException::InvalidVersionException() : Exception("Holofile: Invalid version") {}

InvalidFrameSizeException::InvalidFrameSizeException()
    : Exception("Holofile: Invalid frame size") {}

Reader::Reader(const std::string &path) : frame_index_(0) {
  // Open file
  FILE *fp = nullptr;
  if (fopen_s(&fp, path.c_str(), "rb") != 0 || !fp) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to open \"" + path + "\"");
  }
  file_.reset(fp);

  // Read header
  std::size_t success = fread(&header_, sizeof(header_), 1, file_.get());
  if (ferror(file_.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to read header:");
  }
  if (feof(file_.get()) != 0) {
    throw IncompleteHeaderException();
  }

  if (!success) {
    logger()->critical("Unrecoverable error: fread() failed to read the "
                       "header");
    std::exit(EXIT_FAILURE);
  }

  // Check the header
  uint32_t magic_number = std::endian::native == std::endian::little ? 0x4F4C4F48 : 0x484F4C4F;
  if (header_.magic_number != magic_number) {
    throw InvalidMagicNumberException();
  }

  // TODO: Version support.

  std::size_t pixel_per_frame = header_.frame_width * header_.frame_height;
  std::size_t bits_per_frame  = pixel_per_frame * header_.bits_per_pixel;
  if (bits_per_frame % 8 != 0) {
    throw InvalidFrameSizeException();
  }
}

const Header &Reader::header() const { return header_; }

void Reader::seek(std::size_t frame_index) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame   = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;

  size_t offset = sizeof(header_) + frame_index * bytes_per_frame;
  if (fseek(file_.get(), static_cast<long>(offset), SEEK_SET) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek:");
  }

  frame_index_ = frame_index;
}

std::size_t Reader::tell() const { return frame_index_; }

void Reader::read_frames(uint8_t *data, std::size_t frame_count) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame   = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;

  size_t frames_read = fread(data, bytes_per_frame, frame_count, file_.get());
  frame_index_ += frames_read;

  if (ferror(file_.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to read frames:");
  }
  if (frames_read != frame_count && feof(file_.get()) != 0) {
    throw EndOfFileException();
  }
  if (frames_read != frame_count) {
    logger()->critical("Unrecoverable error: fread() failed to read the "
                       "requested number of frames.");
    std::exit(EXIT_FAILURE);
  }
}

} // namespace holofile