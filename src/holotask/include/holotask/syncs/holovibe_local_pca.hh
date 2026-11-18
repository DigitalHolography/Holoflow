#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include "holoflow/core/tasks.hh"

namespace holotask::syncs {
struct HolovibeLocalPcaSettings {
  int width_x = 32;
  int width_y = 32;
  int stride_x = 16;
  int stride_y = 16;
  float rank_factor = 1.f;
  bool operator==(const HolovibeLocalPcaSettings &) const = default;
};

void to_json(nlohmann::json &j, const HolovibeLocalPcaSettings &s);
void from_json(const nlohmann::json &j, HolovibeLocalPcaSettings &s);

class HolovibeLocalPcaFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask> create(
      std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
      const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask> update(
      std::unique_ptr<holoflow::core::ISyncTask> old_task,
      std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
      const holoflow::core::SyncCreateCtx &ctx) const override;
};
} // namespace holotask::syncs
