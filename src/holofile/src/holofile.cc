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

InvalidFooterException::InvalidFooterException() : Exception("Holofile: Invalid footer") {}

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

  try {
    read_footer();
  } catch (const Exception &e) {
    logger()->warn("Holofile footer could not be read: {}", e.what());
    footer_ = std::nullopt;

    if (fseek(file_.get(), sizeof(header_), SEEK_SET) != 0) {
      std::error_code ec(errno, std::generic_category());
      throw std::system_error(ec, "Failed to seek:");
    }
  }
}

const Header &Reader::header() const { return header_; }

std::optional<Footer> Reader::footer() const { return footer_; }

void Reader::read_footer() {
  size_t footer_offset = sizeof(Header) + header_.data_size_in_bytes;

  int64_t current_pos = _ftelli64(file_.get());
  if (current_pos == -1) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to get current file position:");
  }

  if (_fseeki64(file_.get(), 0, SEEK_END) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek to end:");
  }
  int64_t file_size = _ftelli64(file_.get());
  if (file_size == -1) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to get file size:");
  }

  if (static_cast<size_t>(file_size) <= footer_offset) {
    logger()->info("No footer found - file ends at data section");
    throw InvalidFooterException();
  }

  size_t footer_size = file_size - footer_offset;

  if (footer_size > 1024 * 1024) {
    logger()->warn("Footer appears too large ({} bytes), likely not a valid footer", footer_size);
    throw InvalidFooterException();
  }

  if (_fseeki64(file_.get(), static_cast<int64_t>(footer_offset), SEEK_SET) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek to footer:");
  }

  std::string footer_json;
  footer_json.resize(footer_size);

  size_t bytes_read = fread(footer_json.data(), 1, footer_size, file_.get());
  if (bytes_read != footer_size) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to read footer:");
  }

  if (footer_json.empty() || footer_json[0] != '{') {
    logger()->warn("Footer does not appear to be valid JSON (starts with '{}')",
                   footer_json.empty() ? "empty" : std::string(1, footer_json[0]));
    throw InvalidFooterException();
  }

  try {
    Footer footer;
    footer.pipeline_settings = nlohmann::json::parse(footer_json);
    footer_                  = footer;
  } catch (const nlohmann::json::exception &e) {
    logger()->error("Failed to parse Holofile footer JSON: {}", e.what());
    std::string preview = footer_json.substr(0, std::min(footer_json.size(), size_t(100)));
    logger()->debug("Footer content preview: {}", preview);
    throw InvalidFooterException();
  }

  if (_fseeki64(file_.get(), current_pos, SEEK_SET) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to restore file position:");
  }
}

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

Writer::Writer(const std::string &path, const Header &header, const Footer &footer)
    : header_(header), frame_index_(0), footer_(footer) {
  // Open file
  FILE *fp = nullptr;
  if (fopen_s(&fp, path.c_str(), "wb") != 0 || !fp) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to open \"" + path + "\"");
  }
  file_.reset(fp);

  // Write header
  std::size_t success = fwrite(&header_, sizeof(header_), 1, file_.get());
  if (ferror(file_.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to write header:");
  }

  if (!success) {
    logger()->critical("Unrecoverable error: fwrite() failed to write the "
                       "header");
    std::exit(EXIT_FAILURE);
  }
}

void Writer::write_footer() {
  std::string footer_json = footer_.pipeline_settings.dump();

  logger()->info("Writing Holofile footer with pipeline settings: {}", footer_json);

  if (_fseeki64(file_.get(), 0, SEEK_END) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek to end of file:");
  }

  if (fwrite(footer_json.data(), 1, footer_json.size(), file_.get()) != footer_json.size()) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to write footer JSON:");
  }

  if (fflush(file_.get()) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to flush file:");
  }
}

void Writer::write_frames(const uint8_t *data, std::size_t frame_count) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame   = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;

  size_t frames_written = fwrite(data, bytes_per_frame, frame_count, file_.get());
  frame_index_ += frames_written;

  if (ferror(file_.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to write frames:");
  }
  if (frames_written != frame_count) {
    logger()->critical("Unrecoverable error: fwrite() failed to write the "
                       "requested number of frames.");
    std::exit(EXIT_FAILURE);
  }
}

size_t Writer::tell() const { return frame_index_; }

} // namespace holofile