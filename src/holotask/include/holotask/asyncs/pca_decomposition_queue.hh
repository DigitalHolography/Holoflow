// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/core/tasks.hh"

namespace holotask::asyncs {

struct PcaDecompositionQueueSettings {
  size_t batch_size    = 4;
  size_t slot_count    = 3;
  bool   use_fused_f16 = false;

  bool operator==(const PcaDecompositionQueueSettings &) const = default;
};

void to_json(nlohmann::json &j, const PcaDecompositionQueueSettings &settings);
void from_json(const nlohmann::json &j, PcaDecompositionQueueSettings &settings);

// Accumulates [1,T,H,W] inputs. Each pop retains the paired [B,T,H,W] input and
// [B,T,T] eigenvectors until the consumer has finished using both outputs.
class PcaDecompositionQueueFactory final : public holoflow::core::IAsyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &settings) const override;

  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &settings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::IAsyncTask>
  update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &settings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;
};

} // namespace holotask::asyncs
