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

#include "holonp/asarray.hh"

#include <stdexcept>

namespace holonp {

// -------------------------------------------------------------------------------------------------
// AsArray JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const AsArraySettings &s) {
  j = nlohmann::json{{"value", s.value}};
  if (s.device) {
    j["device"] = *s.device;
  }
}

void from_json(const nlohmann::json &j, AsArraySettings &s) {
  j.at("value").get_to(s.value);
  if (j.contains("device")) {
    s.device = j.at("device").get<holoflow::core::MemLoc>();
  } else {
    s.device = std::nullopt;
  }
}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("AsArrayFactory inference error: " + msg);
  }
}

// -------------------------------------------------------------------------------------------------
// Task implementation
// -------------------------------------------------------------------------------------------------

class AsArray : public holoflow::core::ISyncTask {
public:
  explicit AsArray(AsArraySettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const AsArraySettings &settings() const { return settings_; }
  void                   update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  AsArraySettings settings_;
  cudaStream_t    stream_;
};

} // namespace

holoflow::core::OpResult AsArray::execute(holoflow::core::SyncCtx &ctx) {
  auto *odata = ctx.outputs[0].data();
  auto  odesc = ctx.outputs[0].desc;

  if (odesc.dtype != holoflow::core::DType::F32) {
    logger()->error("[AsArray::execute] unsupported dtype");
    std::abort();
  }

  const float value = static_cast<float>(settings_.value);
  CUDA_CHECK(cudaMemcpyAsync(odata, &value, sizeof(float), cudaMemcpyHostToDevice, stream_));
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
AsArrayFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<AsArraySettings>();
  const auto memloc   = settings.device.value_or(holoflow::core::MemLoc::Device);

  check(input_descs.empty(), "expected zero inputs");
  check(memloc == holoflow::core::MemLoc::Device, "only Device output is supported (for now)");

  holoflow::core::TDesc odesc({1}, holoflow::core::DType::F32, memloc);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AsArrayFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  auto settings = jsettings.get<AsArraySettings>();

  return std::make_unique<AsArray>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
AsArrayFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_asarray = dynamic_cast<AsArray *>(old_task.get());
  if (old_asarray == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings = jsettings.get<AsArraySettings>();
  if (settings == old_asarray->settings()) {
    old_asarray->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
