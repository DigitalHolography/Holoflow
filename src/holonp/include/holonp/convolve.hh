// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#pragma once

#include <nlohmann/json.hpp>
#include <span>
#include <string>

#include "holoflow/core/tasks.hh"

namespace holonp {

struct ConvolveSettings {
  std::string mode = "full";

  bool operator==(const ConvolveSettings &) const = default;
};

void to_json(nlohmann::json &j, const ConvolveSettings &s);
void from_json(const nlohmann::json &j, ConvolveSettings &s);

class ConvolveFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holonp
