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

#include "holonp/zeros.hh"

#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const ZerosSettings &s) {
  j = nlohmann::json{
      {"shape", s.shape},
      {"order", s.order},
  };

  if (s.dtype) {
    j["dtype"] = *s.dtype;
  }
  if (s.device) {
    j["device"] = *s.device;
  }
}

void from_json(const nlohmann::json &j, ZerosSettings &s) {
  j.at("shape").get_to(s.shape);

  if (j.contains("order")) {
    j.at("order").get_to(s.order);
  } else {
    s.order = "C";
  }

  if (j.contains("dtype")) {
    s.dtype = j.at("dtype").get<holoflow::core::DType>();
  } else {
    s.dtype = std::nullopt;
  }

  if (j.contains("device")) {
    s.device = j.at("device").get<holoflow::core::MemLoc>();
  } else {
    s.device = std::nullopt;
  }
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("ZerosFactory inference error: " + msg);
  }
}

size_t total_elements(const std::vector<size_t> &shape) {
  size_t total = 1;
  for (auto dim : shape) {
    total *= dim;
  }
  return total;
}

} // namespace

Zeros::Zeros(const ZerosSettings &settings, holoflow::core::DType dtype, size_t total_elems,
             cudaStream_t stream)
    : settings_(settings), dtype_(dtype), total_elems_(total_elems), stream_(stream) {}

holoflow::core::OpResult Zeros::execute(holoflow::core::SyncCtx &ctx) {
  auto bytes = total_elems_ * holoflow::core::size_of(dtype_);
  if (bytes == 0)
    return holoflow::core::OpResult::Ok;

  CUDA_CHECK(cudaMemsetAsync(ctx.outputs[0].data(), 0, bytes, stream_));
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult ZerosFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<ZerosSettings>();
  const auto dtype    = settings.dtype.value_or(holoflow::core::DType::F32);
  const auto memloc   = settings.device.value_or(holoflow::core::MemLoc::Device);

  check(input_descs.empty(), "expected zero inputs");
  check(settings.order == "C", "only C order is supported");
  check(memloc == holoflow::core::MemLoc::Device, "only Device output is supported (for now)");

  holoflow::core::TDesc odesc({settings.shape}, dtype, memloc);

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
ZerosFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                     const nlohmann::json                  &jsettings,
                     const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<ZerosSettings>();
  auto dtype    = settings.dtype.value_or(holoflow::core::DType::F32);
  (void)infer;

  auto total = total_elements(settings.shape);
  return std::unique_ptr<holoflow::core::ISyncTask>(new Zeros(settings, dtype, total, ctx.stream));
}

} // namespace holonp
