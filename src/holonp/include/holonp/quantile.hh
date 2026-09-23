#pragma once

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holonp {

struct QuantileSettings {
  float            q = 0.5f;
  std::vector<int> axis;
  bool             keepdims = false;

  bool operator==(const QuantileSettings &) const = default;
};

void to_json(nlohmann::json &j, const QuantileSettings &settings);
void from_json(const nlohmann::json &j, QuantileSettings &settings);

class QuantileFactory : public holoflow::core::ISyncTaskFactory {
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
