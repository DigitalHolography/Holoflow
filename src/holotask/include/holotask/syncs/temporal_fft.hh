// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "holoflow/core/tasks.hh"

namespace holotask::syncs {
/// Unnormalized complex FFT along T of a contiguous device tensor [T,Y,X].
/// Matches Holovibes STFT: rectangular window, no temporal FFT shift.
class TemporalFFTFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc>,
         const nlohmann::json &, const holoflow::core::SyncCreateCtx &) const override;
};
} // namespace holotask::syncs
