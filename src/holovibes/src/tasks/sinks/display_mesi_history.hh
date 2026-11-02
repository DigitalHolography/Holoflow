// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <QPointer>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>

#include "holoflow/core/tasks.hh"
#include "holotask/syncs/mesi_remi.hh"

namespace holovibes::ui {
class MesiDashboardWidget;
}

namespace holovibes::tasks::sinks {

struct DisplayMesiHistorySettings {
  double                            tau_min_seconds = 0.0;
  double                            tau_max_seconds = 1.0;
  holotask::syncs::MesiRemiSettings model_settings;
  size_t                            source_cycle_packets = 0;
  std::string                       curve_cache_key;
};

void to_json(nlohmann::json &j, const DisplayMesiHistorySettings &settings);
void from_json(const nlohmann::json &j, DisplayMesiHistorySettings &settings);

class DisplayMesiHistoryFactory : public holoflow::core::ISyncTaskFactory {
public:
  explicit DisplayMesiHistoryFactory(holovibes::ui::MesiDashboardWidget *dashboard_widget);

  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

private:
  QPointer<holovibes::ui::MesiDashboardWidget> dashboard_widget_;
};

} // namespace holovibes::tasks::sinks
