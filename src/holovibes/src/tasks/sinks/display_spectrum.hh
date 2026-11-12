// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/core/tasks.hh"

#include <nlohmann/json.hpp>

namespace holovibes::ui {
class SpectrumWidget;
}

namespace holovibes::tasks::sinks {

struct DisplaySpectrumSettings {
  size_t window_size           = 1;
  double sampling_frequency_hz = 1.0;
  bool   full_fft              = false;
};

void to_json(nlohmann::json &j, const DisplaySpectrumSettings &settings);
void from_json(const nlohmann::json &j, DisplaySpectrumSettings &settings);

class DisplaySpectrumFactory : public holoflow::core::ISyncTaskFactory {
public:
  explicit DisplaySpectrumFactory(ui::SpectrumWidget *widget);

  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

private:
  ui::SpectrumWidget *widget_;
};

} // namespace holovibes::tasks::sinks
