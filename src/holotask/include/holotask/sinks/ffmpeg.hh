// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "holoflow/core/tasks.hh"

namespace holotask::sinks {

struct FfmpegSettings {
  std::string path;
  int         count;
  double      fps;
  std::string format;
  std::string codec;

  bool operator==(const FfmpegSettings &) const = default;
};

void to_json(nlohmann::json &j, const FfmpegSettings &settings);
void from_json(const nlohmann::json &j, FfmpegSettings &settings);

class FfmpegFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holotask::sinks
