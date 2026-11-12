// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

enum class LocalPcaMap { Energy, Original, Static, Dynamic };

struct LocalPcaSettings {
  int window_size = 5;
  int stride = 1;
  int begin = 0;
  int end = 15;
  bool automatic = false;
  LocalPcaMap map = LocalPcaMap::Energy;

  bool operator==(const LocalPcaSettings &) const = default;
};

void to_json(nlohmann::json &j, const LocalPcaSettings &s);
void from_json(const nlohmann::json &j, LocalPcaSettings &s);

class LocalPcaFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                   const nlohmann::json &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
