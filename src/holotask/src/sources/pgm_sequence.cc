// Copyright 2026 Digital Holography Foundation
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

#include "holotask/sources/pgm_sequence.hh"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "holoflow/core/tensor.hh"

namespace holotask::sources {

void to_json(nlohmann::json &j, const PgmSequenceSettings &settings) {
  j = {{"directory", settings.directory},
       {"filename_prefix", settings.filename_prefix},
       {"batch_size", settings.batch_size},
       {"repeat", settings.repeat},
       {"expected_count", settings.expected_count}};
}

void from_json(const nlohmann::json &j, PgmSequenceSettings &settings) {
  j.at("directory").get_to(settings.directory);
  settings.filename_prefix = j.value("filename_prefix", std::string{"i"});
  settings.batch_size      = j.value("batch_size", 1);
  settings.repeat          = j.value("repeat", true);
  settings.expected_count  = j.value("expected_count", 0);
}

namespace {

struct PgmHeader {
  std::size_t  width;
  std::size_t  height;
  unsigned int max_value;
  bool         binary;
};

std::string read_token(std::istream &input, const std::filesystem::path &path) {
  while (true) {
    const int next = input.peek();
    if (next == std::char_traits<char>::eof()) {
      throw std::runtime_error("Unexpected end of PGM header in " + path.string());
    }
    if (std::isspace(static_cast<unsigned char>(next))) {
      input.get();
      continue;
    }
    if (next == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    break;
  }

  std::string token;
  while (true) {
    const int next = input.peek();
    if (next == std::char_traits<char>::eof() || std::isspace(static_cast<unsigned char>(next)) ||
        next == '#') {
      break;
    }
    token.push_back(static_cast<char>(input.get()));
  }
  if (token.empty()) {
    throw std::runtime_error("Invalid PGM token in " + path.string());
  }
  return token;
}

template <typename T> T parse_integer(const std::string &token, const std::filesystem::path &path) {
  T value{};
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size()) {
    throw std::runtime_error("Invalid integer '" + token + "' in " + path.string());
  }
  return value;
}

PgmHeader read_header(std::istream &input, const std::filesystem::path &path) {
  const auto magic = read_token(input, path);
  if (magic != "P2" && magic != "P5") {
    throw std::runtime_error("Unsupported PGM format '" + magic + "' in " + path.string() +
                             " (expected P2 or P5)");
  }

  const auto width  = parse_integer<std::size_t>(read_token(input, path), path);
  const auto height = parse_integer<std::size_t>(read_token(input, path), path);
  const auto maxval = parse_integer<unsigned int>(read_token(input, path), path);
  if (width == 0 || height == 0 || maxval == 0 || maxval > 65535 ||
      width > std::numeric_limits<std::size_t>::max() / height) {
    throw std::runtime_error("Invalid PGM dimensions or maximum value in " + path.string());
  }

  if (magic == "P5") {
    const int separator = input.get();
    if (separator == std::char_traits<char>::eof() ||
        !std::isspace(static_cast<unsigned char>(separator))) {
      throw std::runtime_error("Missing PGM raster separator in " + path.string());
    }
    if (separator == '\r' && input.peek() == '\n') {
      input.get();
    }
  }

  return {width, height, maxval, magic == "P5"};
}

std::vector<std::filesystem::path> find_frames(const PgmSequenceSettings &settings) {
  const std::filesystem::path directory(settings.directory);
  if (!std::filesystem::is_directory(directory)) {
    throw std::invalid_argument("PGM sequence directory does not exist: " + directory.string());
  }

  std::vector<std::filesystem::path> paths;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pgm") {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.compare(0, settings.filename_prefix.size(), settings.filename_prefix) == 0) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    throw std::invalid_argument("No PGM frames starting with '" + settings.filename_prefix +
                                "' found in " + directory.string());
  }
  return paths;
}

struct SequenceInfo {
  std::vector<std::filesystem::path> paths;
  PgmHeader                          header;
  holoflow::core::DType              dtype;
};

SequenceInfo inspect_sequence(const PgmSequenceSettings &settings) {
  auto paths = find_frames(settings);
  if (settings.expected_count < 0 ||
      (settings.expected_count > 0 && paths.size() != static_cast<size_t>(settings.expected_count)))
    throw std::invalid_argument("PGM sequence file count differs from expected_count");
  std::optional<PgmHeader> first_header;
  for (const auto &path : paths) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Could not open PGM frame: " + path.string());
    }
    const auto header = read_header(input, path);
    if (first_header.has_value() &&
        (header.width != first_header->width || header.height != first_header->height ||
         (header.max_value <= 255) != (first_header->max_value <= 255))) {
      throw std::runtime_error("PGM frames have inconsistent dimensions or bit depth: " +
                               path.string());
    }
    first_header = header;
  }

  if (settings.batch_size <= 0 || static_cast<std::size_t>(settings.batch_size) > paths.size()) {
    throw std::invalid_argument("PGM sequence batch_size must be positive and no larger than "
                                "the number of matching frames");
  }
  if (!settings.repeat && paths.size() % static_cast<std::size_t>(settings.batch_size) != 0) {
    throw std::invalid_argument("A finite PGM sequence requires complete batches; batch_size must "
                                "divide the number of frames");
  }

  const auto dtype =
      first_header->max_value <= 255 ? holoflow::core::DType::U8 : holoflow::core::DType::U16;
  const auto pixels_per_frame = first_header->width * first_header->height;
  const auto bytes_per_pixel  = holoflow::core::size_of(dtype);
  if (pixels_per_frame > std::numeric_limits<std::size_t>::max() / bytes_per_pixel ||
      paths.size() >
          std::numeric_limits<std::size_t>::max() / (pixels_per_frame * bytes_per_pixel)) {
    throw std::invalid_argument("PGM sequence is too large to fit in host memory");
  }
  return {std::move(paths), *first_header, dtype};
}

void read_frame(const std::filesystem::path &path, const PgmHeader &expected,
                holoflow::core::DType dtype, std::byte *output) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Could not open PGM frame: " + path.string());
  }
  const auto header = read_header(input, path);
  if (header.width != expected.width || header.height != expected.height ||
      (header.max_value <= 255) != (expected.max_value <= 255)) {
    throw std::runtime_error("PGM frame changed after sequence inspection: " + path.string());
  }

  const auto pixel_count = header.width * header.height;
  if (header.binary) {
    const auto raster_bytes = pixel_count * holoflow::core::size_of(dtype);
    if (raster_bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
      throw std::runtime_error("PGM raster exceeds supported stream size: " + path.string());
    input.read(reinterpret_cast<char *>(output), static_cast<std::streamsize>(raster_bytes));
    if (static_cast<std::size_t>(input.gcount()) != raster_bytes) {
      throw std::runtime_error("Truncated PGM raster in " + path.string());
    }
    if (dtype == holoflow::core::DType::U16) {
      auto *pixels = reinterpret_cast<std::uint16_t *>(output);
      for (std::size_t i = 0; i < pixel_count; ++i) {
        const auto *bytes = output + 2 * i;
        const auto  value =
            (static_cast<unsigned int>(std::to_integer<unsigned char>(bytes[0])) << 8) |
            std::to_integer<unsigned char>(bytes[1]);
        if (value > header.max_value) {
          throw std::runtime_error("PGM pixel exceeds maxval in " + path.string());
        }
        pixels[i] = static_cast<std::uint16_t>(value);
      }
    } else if (header.max_value < 255) {
      for (std::size_t i = 0; i < pixel_count; ++i) {
        if (std::to_integer<unsigned int>(output[i]) > header.max_value) {
          throw std::runtime_error("PGM pixel exceeds maxval in " + path.string());
        }
      }
    }
    return;
  }
  for (std::size_t i = 0; i < pixel_count; ++i) {
    const auto value = parse_integer<unsigned int>(read_token(input, path), path);

    if (value > header.max_value) {
      throw std::runtime_error("PGM pixel exceeds maxval in " + path.string());
    }
    if (dtype == holoflow::core::DType::U8) {
      reinterpret_cast<std::uint8_t *>(output)[i] = static_cast<std::uint8_t>(value);
    } else {
      reinterpret_cast<std::uint16_t *>(output)[i] = static_cast<std::uint16_t>(value);
    }
  }
}

class PgmSequence : public holoflow::core::ISyncTask {
public:
  PgmSequence(PgmSequenceSettings settings, SequenceInfo sequence,
              holoflow::core::TDesc output_desc)
      : settings_(std::move(settings)), sequence_(std::move(sequence)),
        output_desc_(std::move(output_desc)),
        bytes_per_frame_(sequence_.header.width * sequence_.header.height *
                         holoflow::core::size_of(sequence_.dtype)),
        frames_(sequence_.paths.size() * bytes_per_frame_) {
    for (std::size_t frame = 0; frame < sequence_.paths.size(); ++frame) {
      read_frame(sequence_.paths[frame], sequence_.header, sequence_.dtype,
                 frames_.data() + frame * bytes_per_frame_);
    }
    storage_ = {holoflow::core::MemLoc::Host, frames_.size(), frames_.data()};
  }

  bool has_settings(const PgmSequenceSettings &settings) const { return settings_ == settings; }

  std::optional<holoflow::core::TView> acquire_input(int) override {
    throw std::out_of_range("PGM sequence source has no inputs");
  }

  void release_output(int index) override {
    if (index != 0) {
      throw std::out_of_range("PGM sequence source has only one output at index 0");
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (ctx.cancelled != nullptr && ctx.cancelled->load(std::memory_order_relaxed)) {
      return holoflow::core::OpResult::Cancelled;
    }
    const auto batch_size = static_cast<std::size_t>(settings_.batch_size);
    if (next_frame_ + batch_size > sequence_.paths.size()) {
      if (!settings_.repeat) {
        return holoflow::core::OpResult::Eof;
      }
      next_frame_ = 0;
    }

    // Preserve the compiler's shared Storage identity so downstream views see the
    // published pointer. Direct factory users may supply an empty output slot.
    auto *storage  = ctx.outputs[0].storage != nullptr ? ctx.outputs[0].storage : &storage_;
    *storage       = {holoflow::core::MemLoc::Host, batch_size * bytes_per_frame_,
                      frames_.data() + next_frame_ * bytes_per_frame_};
    ctx.outputs[0] = {.desc = output_desc_, .storage = storage};
    next_frame_ += batch_size;
    return holoflow::core::OpResult::Ok;
  }

private:
  PgmSequenceSettings     settings_;
  SequenceInfo            sequence_;
  holoflow::core::TDesc   output_desc_;
  std::size_t             bytes_per_frame_;
  std::vector<std::byte>  frames_;
  holoflow::core::Storage storage_{};
  std::size_t             next_frame_ = 0;
};

} // namespace

holoflow::core::InferResult
PgmSequenceFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings) const {
  if (!input_descs.empty()) {
    throw std::invalid_argument("PGM sequence source must have no inputs");
  }

  const auto                  settings = jsettings.get<PgmSequenceSettings>();
  const auto                  sequence = inspect_sequence(settings);
  const holoflow::core::TDesc output_desc({static_cast<std::size_t>(settings.batch_size),
                                           sequence.header.height, sequence.header.width},
                                          sequence.dtype, holoflow::core::MemLoc::Host);

  return {.input_descs   = {},
          .output_descs  = {output_desc},
          .in_place      = {},
          .owned_inputs  = {},
          .owned_outputs = {true},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
PgmSequenceFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings,
                           const holoflow::core::SyncCreateCtx &) const {
  if (!input_descs.empty()) {
    throw std::invalid_argument("PGM sequence source must have no inputs");
  }
  const auto                  settings = jsettings.get<PgmSequenceSettings>();
  const auto                  sequence = inspect_sequence(settings);
  const holoflow::core::TDesc output_desc({static_cast<std::size_t>(settings.batch_size),
                                           sequence.header.height, sequence.header.width},
                                          sequence.dtype, holoflow::core::MemLoc::Host);
  return std::make_unique<PgmSequence>(settings, sequence, output_desc);
}

std::unique_ptr<holoflow::core::ISyncTask>
PgmSequenceFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                           std::span<const holoflow::core::TDesc>     input_descs,
                           const nlohmann::json                      &jsettings,
                           const holoflow::core::SyncCreateCtx       &ctx) const {
  const auto settings = jsettings.get<PgmSequenceSettings>();
  if (input_descs.empty()) {
    if (const auto *pgm_sequence = dynamic_cast<const PgmSequence *>(old_task.get());
        pgm_sequence != nullptr && pgm_sequence->has_settings(settings)) {
      return old_task;
    }
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources
