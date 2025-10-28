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

#include "holoflow/core/tasks.hh"

#include <optional>

#include "bug.hh"
#include "holoflow/core/tensor.hh"

namespace holoflow::core {

std::optional<TView> ITask::acquire_input(int) {
  throw std::out_of_range("Input index out of range");
}

void ITask::release_output(int) { throw std::out_of_range("Output index out of range"); }

void ITask::bind_logger(std::shared_ptr<spdlog::logger> logger) {
  HOLOFLOW_CHECK(logger != nullptr, "Cannot bind null logger to task");
  logger_ = std::move(logger);
}

spdlog::logger *ITask::logger() {
  HOLOFLOW_CHECK(logger_, "Logger not bound to task");
  return logger_.get();
}

std::unique_ptr<ISyncTask> ISyncTaskFactory::update(std::unique_ptr<ISyncTask>,
                                                    std::span<const TDesc> input_descs,
                                                    const nlohmann::json  &jsettings,
                                                    const SyncCreateCtx   &ctx) const {
  return create(input_descs, jsettings, ctx);
}

std::unique_ptr<IAsyncTask> IAsyncTaskFactory::update(std::unique_ptr<IAsyncTask>,
                                                      std::span<const TDesc> input_descs,
                                                      const nlohmann::json  &jsettings,
                                                      const AsyncCreateCtx  &ctx) const {
  return create(input_descs, jsettings, ctx);
}

} // namespace holoflow::core