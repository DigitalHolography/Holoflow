#pragma once

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holonp {

struct MedianSettings {
  std::vector<int> axis;
  bool             keepdims = false;

  bool operator==(const MedianSettings &) const = default;
};

void to_json(nlohmann::json &j, const MedianSettings &settings);
void from_json(const nlohmann::json &j, MedianSettings &settings);

class MedianFactory : public holoflow::core::ISyncTaskFactory {
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
