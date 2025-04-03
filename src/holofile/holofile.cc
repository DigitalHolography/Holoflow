#include "holofile/holofile.hh"

#include <bit>
#include <iostream>

namespace dh {

const char *holofile_error_category::name() const noexcept {
  return "holofile";
}

std::string holofile_error_category::message(int ev) const {
  switch (static_cast<HolofileError>(ev)) {
  case HolofileError::EndOfFile:
    return "End of file";
  case HolofileError::IncompleteHeader:
    return "Incomplete header";
  case HolofileError::InvalidMagicNumber:
    return "Invalid magic number";
  case HolofileError::InvalidVersion:
    return "Invalid version";
  case HolofileError::InvalidFrameSize:
    return "Invalid frame size";
  default:
    return "Unknown error";
  }
}

const std::error_category &holofile_category() {
  static holofile_error_category instance;
  return instance;
}

std::error_code make_error_code(HolofileError e) {
  return {static_cast<int>(e), holofile_category()};
}

tl::expected<HolofileReader, std::error_code>
HolofileReader::open(const std::string &filename) {
  // Open the file for reading.
  FILE *fp = nullptr;
  if (fopen_s(&fp, filename.c_str(), "rb") != 0 || !fp)
    return tl::unexpected(std::error_code(errno, std::generic_category()));
  std::unique_ptr<FILE, FileCloser> file(fp);

  // Read the header.
  HolofileHeader header;
  size_t success = fread(&header, sizeof(HolofileHeader), 1, file.get());

  // TODO: Check if one should use clearerr() here.
  if (ferror(file.get()) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  // TODO: Check if one should use clearerr() here.
  if (feof(file.get()) != 0)
    return tl::unexpected(make_error_code(HolofileError::IncompleteHeader));

  if (!success) {
    std::cerr << "[holofile] Unrecoverable error: fread() failed to read the "
                 "header.\n";
    std::exit(EXIT_FAILURE);
  }

  // Check the header.
  uint32_t magic_number =
      std::endian::native == std::endian::little ? 0x4F4C4F48 : 0x484F4C4F;

  if (header.magic_number != magic_number)
    return tl::unexpected(make_error_code(HolofileError::InvalidMagicNumber));

  // TODO version support
  // if (header.version != 7)
  //   return tl::unexpected(make_error_code(HolofileError::InvalidVersion));

  size_t pixels_per_frame = header.frame_width * header.frame_height;
  size_t bits_per_frame = pixels_per_frame * header.bits_per_pixel;
  if (bits_per_frame % 8 != 0)
    return tl::unexpected(make_error_code(HolofileError::InvalidFrameSize));

  return HolofileReader(std::move(file), header);
}

const HolofileHeader &HolofileReader::header() const { return header_; }

tl::expected<void, std::error_code> HolofileReader::seek(size_t frame_index) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame = bits_per_frame / 8;

  size_t offset = sizeof(HolofileHeader) + frame_index * bytes_per_frame;
  if (fseek(file_.get(), static_cast<long>(offset), SEEK_SET) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  frame_index_ = frame_index;
  return {};
}

size_t HolofileReader::tell() const { return frame_index_; }

tl::expected<void, std::error_code> HolofileReader::read_frames(uint8_t *data,
                                                                size_t count) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame = bits_per_frame / 8;
  size_t frames_read = fread(data, bytes_per_frame, count, file_.get());

  frame_index_ += frames_read;

  // TODO: Check if one should use clearerr() here.
  FILE *file = file_.get();
  if (ferror(file) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  // TODO: Check if one should use clearerr() here.
  if (feof(file_.get()) != 0)
    return tl::unexpected(make_error_code(HolofileError::EndOfFile));

  if (frames_read != count) {
    std::cerr << "[holofile] Unrecoverable error: fread() failed to read the "
                 "requested number of frames.\n";
    std::exit(EXIT_FAILURE);
  }

  return {};
}

HolofileReader::HolofileReader(std::unique_ptr<FILE, FileCloser> file,
                               const HolofileHeader &header)
    : file_(std::move(file)), header_(header), frame_index_(0) {}

tl::expected<HolofileWriter, std::error_code>
HolofileWriter::create(const std::string &filename,
                       const HolofileHeader &header) {
  // Open the file for writing.
  FILE *fp = nullptr;
  if (fopen_s(&fp, filename.c_str(), "wb") != 0 || !fp)
    return tl::unexpected(std::error_code(errno, std::generic_category()));
  std::unique_ptr<FILE, FileCloser> file(fp);

  // Write the header.
  size_t written = fwrite(&header, sizeof(HolofileHeader), 1, file.get());
  if (written != 1) {
    if (ferror(file.get()) != 0)
      return tl::unexpected(std::error_code(errno, std::generic_category()));
    std::cerr << "[holofile] Unrecoverable error: fwrite() failed to write the "
                 "header.\n";
    std::exit(EXIT_FAILURE);
  }

  // Flush to ensure the header is written.
  if (fflush(file.get()) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  return HolofileWriter(std::move(file), header);
}

const HolofileHeader &HolofileWriter::header() const { return header_; }

tl::expected<void, std::error_code>
HolofileWriter::write_frames(const uint8_t *data, size_t count) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame = bits_per_frame / 8;
  size_t frames_written = fwrite(data, bytes_per_frame, count, file_.get());
  if (frames_written != count) {
    if (ferror(file_.get()) != 0)
      return tl::unexpected(std::error_code(errno, std::generic_category()));
    std::cerr << "[holofile] Unrecoverable error: fwrite() failed to write the "
                 "requested number of frames.\n";
    std::exit(EXIT_FAILURE);
  }
  frame_index_ += frames_written;

  // Flush to ensure data is written.
  if (fflush(file_.get()) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  return {};
}

HolofileWriter::HolofileWriter(std::unique_ptr<FILE, FileCloser> file,
                               const HolofileHeader &header)
    : file_(std::move(file)), header_(header), frame_index_(0) {}

} // namespace dh