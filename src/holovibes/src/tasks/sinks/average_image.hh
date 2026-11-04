// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "holoflow/core/tasks.hh"

namespace holovibes::tasks::sinks {

struct AverageImageSettings {
  std::string path;
  int         count;
  std::string format;
  bool        output_16bit = false;

  bool operator==(const AverageImageSettings &) const = default;
};

void to_json(nlohmann::json &j, const AverageImageSettings &settings);
void from_json(const nlohmann::json &j, AverageImageSettings &settings);

class AverageImageFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holovibes::tasks::sinks
