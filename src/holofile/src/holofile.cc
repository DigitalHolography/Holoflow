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

#include <bit>
#include <cstdio>
#include <system_error>
#include <utility>

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

// -------------------------------------------------------------------------------------------------
// Private implementation
// -------------------------------------------------------------------------------------------------

namespace {

struct FileCloser {
  void operator()(FILE *file) const { fclose(file); }
};

} // namespace

struct Reader::Impl {
  void read_footer();

  std::unique_ptr<FILE, FileCloser> file;
  Header                            header;
  std::optional<Footer>             footer;
  std::size_t                       frame_index = 0;
};

struct Writer::Impl {
  std::unique_ptr<FILE, FileCloser> file;
  Header                            header;
  Footer                            footer;
  std::size_t                       frame_index = 0;
};

Reader::Reader(const std::string &path) : impl_(std::make_unique<Impl>()) {
  // Open file
  FILE *fp = nullptr;
  if (fopen_s(&fp, path.c_str(), "rb") != 0 || !fp) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to open \"" + path + "\"");
  }
  impl_->file.reset(fp);

  // Read header
  std::size_t success = fread(&impl_->header, sizeof(impl_->header), 1, impl_->file.get());
  if (ferror(impl_->file.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to read header:");
  }
  if (feof(impl_->file.get()) != 0) {
    throw IncompleteHeaderException();
  }

  if (!success) {
    logger()->critical("Unrecoverable error: fread() failed to read the "
                       "header");
    std::exit(EXIT_FAILURE);
  }

  // Check the header
  uint32_t magic_number = std::endian::native == std::endian::little ? 0x4F4C4F48 : 0x484F4C4F;
  if (impl_->header.magic_number != magic_number) {
    throw InvalidMagicNumberException();
  }

  // TODO: Version support.

  std::size_t pixel_per_frame = impl_->header.frame_width * impl_->header.frame_height;
  std::size_t bits_per_frame  = pixel_per_frame * impl_->header.bits_per_pixel;
  if (bits_per_frame % 8 != 0) {
    throw InvalidFrameSizeException();
  }

  try {
    impl_->read_footer();
  } catch (const Exception &e) {
    logger()->warn("Holofile footer could not be read: {}", e.what());
    impl_->footer = std::nullopt;

    if (fseek(impl_->file.get(), sizeof(impl_->header), SEEK_SET) != 0) {
      std::error_code ec(errno, std::generic_category());
      throw std::system_error(ec, "Failed to seek:");
    }
  }
}

Reader::~Reader() = default;

Reader::Reader(Reader &&) noexcept = default;

Reader &Reader::operator=(Reader &&) noexcept = default;

const Header &Reader::header() const { return impl_->header; }

std::optional<Footer> Reader::footer() const { return impl_->footer; }

void Reader::Impl::read_footer() {
  size_t footer_offset = sizeof(Header) + header.data_size_in_bytes;

  int64_t current_pos = _ftelli64(file.get());
  if (current_pos == -1) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to get current file position:");
  }

  if (_fseeki64(file.get(), 0, SEEK_END) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek to end:");
  }
  int64_t file_size = _ftelli64(file.get());
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

  if (_fseeki64(file.get(), static_cast<int64_t>(footer_offset), SEEK_SET) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek to footer:");
  }

  std::string footer_json;
  footer_json.resize(footer_size);

  size_t bytes_read = fread(footer_json.data(), 1, footer_size, file.get());
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
    Footer parsed_footer;
    parsed_footer.pipeline_settings = nlohmann::json::parse(footer_json);
    footer                          = std::move(parsed_footer);
  } catch (const nlohmann::json::exception &e) {
    logger()->error("Failed to parse Holofile footer JSON: {}", e.what());
    std::string preview = footer_json.substr(0, std::min(footer_json.size(), size_t(100)));
    logger()->debug("Footer content preview: {}", preview);
    throw InvalidFooterException();
  }

  if (_fseeki64(file.get(), current_pos, SEEK_SET) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to restore file position:");
  }
}

void Reader::seek(std::size_t frame_index) {
  size_t pixels_per_frame = impl_->header.frame_width * impl_->header.frame_height;
  size_t bits_per_frame   = pixels_per_frame * impl_->header.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;

  size_t offset = sizeof(impl_->header) + frame_index * bytes_per_frame;
  if (_fseeki64(impl_->file.get(), static_cast<int64_t>(offset), SEEK_SET) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek:");
  }

  impl_->frame_index = frame_index;
}

std::size_t Reader::tell() const { return impl_->frame_index; }

void Reader::read_frames(uint8_t *data, std::size_t frame_count) {
  size_t pixels_per_frame = impl_->header.frame_width * impl_->header.frame_height;
  size_t bits_per_frame   = pixels_per_frame * impl_->header.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;

  size_t frames_read = fread(data, bytes_per_frame, frame_count, impl_->file.get());
  impl_->frame_index += frames_read;

  if (ferror(impl_->file.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to read frames:");
  }
  if (frames_read != frame_count && feof(impl_->file.get()) != 0) {
    throw EndOfFileException();
  }
  if (frames_read != frame_count) {
    logger()->critical("Unrecoverable error: fread() failed to read the "
                       "requested number of frames.");
    std::exit(EXIT_FAILURE);
  }
}

Writer::Writer(const std::string &path, const Header &header, const Footer &footer)
    : impl_(std::make_unique<Impl>()) {
  impl_->header = header;
  impl_->footer = footer;
  // Open file
  FILE *fp = nullptr;
  if (fopen_s(&fp, path.c_str(), "wb") != 0 || !fp) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to open \"" + path + "\"");
  }
  impl_->file.reset(fp);

  // Write header
  std::size_t success = fwrite(&impl_->header, sizeof(impl_->header), 1, impl_->file.get());
  if (ferror(impl_->file.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to write header:");
  }

  if (!success) {
    logger()->critical("Unrecoverable error: fwrite() failed to write the "
                       "header");
    std::exit(EXIT_FAILURE);
  }
}

Writer::~Writer() = default;

Writer::Writer(Writer &&) noexcept = default;

Writer &Writer::operator=(Writer &&) noexcept = default;

void Writer::write_footer() {
  std::string footer_json = impl_->footer.pipeline_settings.dump();

  logger()->info("Writing Holofile footer with pipeline settings: {}", footer_json);

  if (_fseeki64(impl_->file.get(), 0, SEEK_END) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to seek to end of file:");
  }

  if (fwrite(footer_json.data(), 1, footer_json.size(), impl_->file.get()) != footer_json.size()) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to write footer JSON:");
  }

  if (fflush(impl_->file.get()) != 0) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to flush file:");
  }
}

void Writer::write_frames(const uint8_t *data, std::size_t frame_count) {
  size_t pixels_per_frame = impl_->header.frame_width * impl_->header.frame_height;
  size_t bits_per_frame   = pixels_per_frame * impl_->header.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;

  size_t frames_written = fwrite(data, bytes_per_frame, frame_count, impl_->file.get());
  impl_->frame_index += frames_written;

  if (ferror(impl_->file.get())) {
    std::error_code ec(errno, std::generic_category());
    throw std::system_error(ec, "Failed to write frames:");
  }
  if (frames_written != frame_count) {
    logger()->critical("Unrecoverable error: fwrite() failed to write the "
                       "requested number of frames.");
    std::exit(EXIT_FAILURE);
  }
}

size_t Writer::tell() const { return impl_->frame_index; }

} // namespace holofile
