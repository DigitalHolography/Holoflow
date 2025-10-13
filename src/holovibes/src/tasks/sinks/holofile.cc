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

#include "holofile.hh"

#include <cstddef>
#include <cstdint>

#include "bug.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holovibes::tasks::sinks {

void to_json(nlohmann::json &j, const HolofileSettings &hs) {
  j = nlohmann::json{
      {"path", hs.path},
      {"count", hs.count},
  };
}

void from_json(const nlohmann::json &j, HolofileSettings &hs) {
  j.at("path").get_to(hs.path);
  j.at("count").get_to(hs.count);
}

Holofile::Holofile(const HolofileSettings &settings, holofile::Writer writer)
    : settings_(settings), writer_(std::move(writer)), frames_written_(0) {}

Holofile::~Holofile() {
  if (frames_written_ == settings_.count) {
    return;
  }

  logger()->warn("[Holofile] Destructor called before all frames were written "
                 "({}/{}), deleting incomplete file at {}",
                 frames_written_, settings_.count, settings_.path);

  // Delete incomplete file
  HOLOVIBES_CHECK(std::remove(settings_.path.c_str()) == 0,
                  "Failed to delete incomplete HoloFile at {}", settings_.path);
}

[[nodiscard]] holoflow::core::OpResult Holofile::execute(holoflow::core::SyncCtx &ctx) {
  if (frames_written_ >= settings_.count) {
    return holoflow::core::OpResult::Ok;
  }

  auto  remaining  = settings_.count - frames_written_;
  auto  batch_size = static_cast<int>(ctx.inputs[0].desc.shape[0]);
  auto  to_write   = std::min(remaining, batch_size);
  auto *idata      = reinterpret_cast<const uint8_t *>(ctx.inputs[0].data);
  writer_.write_frames(idata, to_write);
  frames_written_ += to_write;
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
HolofileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[HolofileFactory::infer] error: {}", msg);
      throw std::invalid_argument("HolofileFactory inference error: " + msg);
    }
  };
  std::set<holoflow::core::DType> supported_dtypes = {
      holoflow::core::DType::U8,
      holoflow::core::DType::U16,
  };

  auto settings = jsettings.get<HolofileSettings>();

  // Validate
  check(input_descs.size() == 1, "Holofile task must have exactly one output");
  auto &idesc = input_descs[0];
  check(idesc.shape.size() == 3, "Output tensor must have rank 3");
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "Output tensor must be in Host memory");
  check(supported_dtypes.contains(idesc.dtype),
        "Unsupported input dtype: " + std::to_string(static_cast<int>(idesc.dtype)));

  // Success
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx &) const {
  std::map<holoflow::core::DType, uint8_t> dtype_to_bpp = {
      {holoflow::core::DType::U8, static_cast<uint8_t>(8)},
      {holoflow::core::DType::U16, static_cast<uint8_t>(16)},
  };

  // Validate
  auto  infer    = this->infer(input_descs, jsettings);
  auto  settings = jsettings.get<HolofileSettings>();
  auto &idesc    = input_descs[0];

  // Setup writer
  auto dtype        = idesc.dtype;
  auto bpp          = dtype_to_bpp.at(dtype);
  auto frame_width  = static_cast<uint32_t>(idesc.shape[2]);
  auto frame_height = static_cast<uint32_t>(idesc.shape[1]);
  auto frame_count  = static_cast<uint32_t>(settings.count);
  auto data_size    = frame_count * frame_height * frame_width * bpp / 8;
  auto header       = holofile::Header{
            .magic_number       = holofile::Header::MAGIC_NUMBER_LE,
            .version            = holofile::Header::CURRENT_VERSION,
            .bits_per_pixel     = bpp,
            .frame_width        = frame_width,
            .frame_height       = frame_height,
            .frame_count        = frame_count,
            .data_size_in_bytes = data_size,
            .endianness         = holofile::Header::LITTLE_ENDIAN,
  };

  auto writer = holofile::Writer(settings.path, header);

  // Success
  auto *task = new Holofile(settings, std::move(writer));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks::sinks