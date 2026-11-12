// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

enum class PcaLsciMap { Original, Static, Dynamic, PcaFrame };

struct PcaLsciSettings {
  int window_size = 5;
  PcaLsciMap map = PcaLsciMap::Dynamic;

  bool operator==(const PcaLsciSettings &) const = default;
};

void to_json(nlohmann::json &j, const PcaLsciSettings &s);
void from_json(const nlohmann::json &j, PcaLsciSettings &s);

class PcaLsciFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                   const nlohmann::json &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
