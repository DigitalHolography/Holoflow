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

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace holofile {

/// @brief Base exception class for all Holofile-related errors.
///
/// This class serves as the root of the exception hierarchy for Holofile.
/// All specific Holofile errors derive from this class. It inherits from
/// std::runtime_error to hold an error message.
class Exception : public std::runtime_error {
public:
  explicit Exception(const std::string &message);
  ~Exception() noexcept override;
};

/// @brief Exception indicating that the end of the file has been reached.
///
/// Thrown when attempting to read past the last available byte of a Holofile.
class EndOfFileException : public Exception {
public:
  EndOfFileException();
};

/// @brief Exception indicating that the file header is incomplete.
///
/// Thrown when the file does not contain enough bytes to read a full header.
class IncompleteHeaderException : public Exception {
public:
  IncompleteHeaderException();
};

/// @brief Exception indicating that the magic number in the file is invalid.
///
/// Thrown when the file's magic number does not match the expected HOLO
/// identifier.
class InvalidMagicNumberException : public Exception {
public:
  InvalidMagicNumberException();
};

/// @brief Exception indicating that the version number in the file is invalid.
///
/// Thrown when the file's version byte does not match a supported version.
class InvalidVersionException : public Exception {
public:
  InvalidVersionException();
};

/// @brief Exception indicating that the frame size in the file is invalid.
///
/// Thrown when the frame size is not a multiple of the required alignment
/// (e.g., not a multiple of 8 bits).
class InvalidFrameSizeException : public Exception {
public:
  InvalidFrameSizeException();
};

#pragma pack(push, 1)
struct Header {
  uint32_t magic_number;
  uint16_t version;
  uint16_t bits_per_pixel;
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t frame_count;
  uint64_t data_size_in_bytes;
  uint8_t  endianness;
  char     padding[35];
};
#pragma pack(pop)

static_assert(sizeof(Header) == 64, "Holofile header must be 64 bytes");

class Reader {
public:
  explicit Reader(const std::string &path);
  const Header &header() const;
  void          seek(std::size_t frame_index);
  std::size_t   tell() const;
  void          read_frames(uint8_t *data, std::size_t frame_count);

private:
  struct FileCloser {
    void operator()(FILE *file) const { fclose(file); }
  };

  std::unique_ptr<FILE, FileCloser> file_;
  Header                            header_;
  std::size_t                       frame_index_;
};

} // namespace holofile