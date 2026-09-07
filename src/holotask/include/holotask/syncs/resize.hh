#pragma once

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

enum class ResizeInterpolation { Bilinear };

struct ResizeSettings {
  int                 width;
  int                 height;
  ResizeInterpolation interpolation = ResizeInterpolation::Bilinear;

  bool operator==(const ResizeSettings &) const = default;
};

void to_json(nlohmann::json &j, const ResizeInterpolation &interpolation);
void from_json(const nlohmann::json &j, ResizeInterpolation &interpolation);
void to_json(nlohmann::json &j, const ResizeSettings &settings);
void from_json(const nlohmann::json &j, ResizeSettings &settings);

class ResizeFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
