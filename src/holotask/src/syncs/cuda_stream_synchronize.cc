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

#include "holotask/syncs/cuda_stream_synchronize.hh"

#include <stdexcept>
#include <string>

#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const CudaStreamSynchronizeSettings &) { j = nlohmann::json{}; }

void from_json(const nlohmann::json &, CudaStreamSynchronizeSettings &) {}

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[CudaStreamSynchronizeFactory::infer] error: {}", msg);
    throw std::invalid_argument("CudaStreamSynchronizeFactory inference error: " + msg);
  }
}

// -------------------------------------------------------------------------------------------------
// CudaStreamSynchronize task implementation
// -------------------------------------------------------------------------------------------------

class CudaStreamSynchronize : public holoflow::core::ISyncTask {
public:
  explicit CudaStreamSynchronize(cudaStream_t stream) : stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    (void)ctx;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return holoflow::core::OpResult::Ok;
  };

  void         update_stream(cudaStream_t stream) { stream_ = stream; }
  cudaStream_t stream() const { return stream_; }

private:
  cudaStream_t stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// CudaStreamSynchronizeFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
CudaStreamSynchronizeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &jsettings) const {
  (void)jsettings.get<CudaStreamSynchronizeSettings>();
  check(input_descs.size() == 1, "CudaStreamSynchronize task must have exactly one input");

  return holoflow::core::InferResult{
      .input_descs   = {input_descs[0]},
      .output_descs  = {input_descs[0]},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CudaStreamSynchronizeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                     const nlohmann::json                  &jsettings,
                                     const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)this->infer(input_descs, jsettings);
  return std::make_unique<CudaStreamSynchronize>(ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
CudaStreamSynchronizeFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                     std::span<const holoflow::core::TDesc>     input_descs,
                                     const nlohmann::json                      &jsettings,
                                     const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)this->infer(input_descs, jsettings);

  auto *old_sync = dynamic_cast<CudaStreamSynchronize *>(old_task.get());
  if (old_sync == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  old_sync->update_stream(ctx.stream);
  return old_task;
}

} // namespace holotask::syncs
