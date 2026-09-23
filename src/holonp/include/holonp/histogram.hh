#pragma once

#include <nlohmann/json.hpp>
#include <span>

#include "holoflow/core/tasks.hh"

namespace holonp {

struct HistogramSettings {
  int   bins = 10;
  float min  = 0.0f;
  float max  = 1.0f;

  bool operator==(const HistogramSettings &) const = default;
};

void to_json(nlohmann::json &j, const HistogramSettings &settings);
void from_json(const nlohmann::json &j, HistogramSettings &settings);

class HistogramFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
