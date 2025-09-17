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

#include "tasks/memcpy.hh"

#include <omp.h>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks {

void to_json(nlohmann::json &j, const MemcpySettings::Target &t) {
  std::map<MemcpySettings::Target, std::string> t_to_str = {
      {MemcpySettings::Target::Host, "Host"},
      {MemcpySettings::Target::Device, "Device"},
  };

  HOLOVIBES_CHECK(t_to_str.contains(t), "Invalid Target enum value");
  j = t_to_str[t];
}

void from_json(const nlohmann::json &j, MemcpySettings::Target &t) {
  std::map<std::string, MemcpySettings::Target> str_to_t = {
      {"Host", MemcpySettings::Target::Host},
      {"Device", MemcpySettings::Target::Device},
  };

  auto key = j.get<std::string>();
  if (!str_to_t.contains(key)) {
    throw std::invalid_argument("Invalid Target string: " + key);
  }
  t = str_to_t[key];
}

void to_json(nlohmann::json &j, const MemcpySettings &ms) {
  j = nlohmann::json{
      {"target", ms.target},
  };
}

void from_json(const nlohmann::json &j, MemcpySettings &ms) { j.at("target").get_to(ms.target); }

namespace {

void mt_memcpy(void *dst, const void *src, const std::size_t n) {
  constexpr int NUM_THREADS = 2;
  auto         *dst_bytes   = static_cast<std::uint8_t *>(dst);
  const auto   *src_bytes   = static_cast<const std::uint8_t *>(src);

  const std::size_t chunk_size = n / NUM_THREADS;
  const std::size_t remainder  = n % NUM_THREADS;

#pragma omp parallel num_threads(NUM_THREADS)
  {
    const int         tid       = omp_get_thread_num();
    const std::size_t offset    = tid * chunk_size;
    std::size_t       this_size = chunk_size;

    if (tid == NUM_THREADS - 1) {
      this_size += remainder;
    }

    if (this_size > 0) {
      std::memcpy(dst_bytes + offset, src_bytes + offset, this_size);
    }
  }
}

} // namespace

Memcpy::Memcpy(const MemcpySettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Memcpy::execute(holoflow::core::SyncCtx &ctx) {
  auto *src       = ctx.inputs[0].data;
  auto *dst       = ctx.outputs[0].data;
  auto  n         = ctx.outputs[0].desc.num_bytes();
  auto  copy_desc = std::make_pair(ctx.inputs[0].desc.mem_loc, settings_.target);
  using CopyDesc  = std::pair<holoflow::core::MemLoc, MemcpySettings::Target>;
  const std::map<CopyDesc, cudaMemcpyKind> copy_map = {
      {{holoflow::core::MemLoc::Host, MemcpySettings::Target::Host}, cudaMemcpyHostToHost},
      {{holoflow::core::MemLoc::Host, MemcpySettings::Target::Device}, cudaMemcpyHostToDevice},
      {{holoflow::core::MemLoc::Device, MemcpySettings::Target::Host}, cudaMemcpyDeviceToHost},
      {{holoflow::core::MemLoc::Device, MemcpySettings::Target::Device}, cudaMemcpyDeviceToDevice},
  };

  HOLOVIBES_CHECK(copy_map.contains(copy_desc), "Invalid memory copy descriptor");
  auto kind = copy_map.at(copy_desc);
  if (kind == cudaMemcpyHostToHost) {
    mt_memcpy(dst, src, n);
  } else {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, n, kind, stream_));
  }

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult MemcpyFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[MemcpyFactory::infer] error: {}", msg);
      throw std::invalid_argument("MemcpyFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<MemcpySettings>();

  // Validate
  check(input_descs.size() == 1, "Memcpy task must have exactly one input");

  // Success
  holoflow::core::TDesc out_desc = input_descs[0];
  out_desc.mem_loc               = settings.target == MemcpySettings::Target::Device
                                       ? holoflow::core::MemLoc::Device
                                       : holoflow::core::MemLoc::Host;

  return holoflow::core::InferResult{
      .input_descs   = {input_descs[0]},
      .output_descs  = {out_desc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
MemcpyFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  this->infer(input_descs, jsettings);
  auto settings = jsettings.get<MemcpySettings>();

  // Success
  auto *task = new Memcpy(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks