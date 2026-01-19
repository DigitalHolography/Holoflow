// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef HOLOTASK_HAS_EGRABBER
#define NOMINMAX
#include <EGrabber.h>
#include <nlohmann/json.hpp>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sources {

struct AmetekS711EuresysCoaxlinkQSFPSettings {
  std::string cfg_path;
};

void to_json(nlohmann::json &j, const AmetekS711EuresysCoaxlinkQSFPSettings &s);
void from_json(const nlohmann::json &j, AmetekS711EuresysCoaxlinkQSFPSettings &s);

class AmetekS711EuresysCoaxlinkQSFP : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  AmetekS711EuresysCoaxlinkQSFP(const AmetekS711EuresysCoaxlinkQSFPSettings &settings,
                                HostPtr<uint8_t>                           &&buffers,
                                std::unique_ptr<Euresys::EGenTL>           &&gentl,
                                std::unique_ptr<Euresys::EGrabber<>>       &&grabber,
                                nlohmann::json                              &cfg);

  friend class AmetekS711EuresysCoaxlinkQSFPFactory;

  AmetekS711EuresysCoaxlinkQSFPSettings settings_;
  HostPtr<uint8_t>                      buffers_;
  std::unique_ptr<Euresys::EGenTL>      gentl_;
  std::unique_ptr<Euresys::EGrabber<>>  grabber_;
  bool                                  running_;
  nlohmann::json                        cfg_;
};

class AmetekS711EuresysCoaxlinkQSFPFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holotask::sources

#else

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::sources {

struct AmetekS711EuresysCoaxlinkQSFPSettings {
  std::string cfg_path;
};

void to_json(nlohmann::json &j, const AmetekS711EuresysCoaxlinkQSFPSettings &s);
void from_json(const nlohmann::json &j, AmetekS711EuresysCoaxlinkQSFPSettings &s);

class AmetekS711EuresysCoaxlinkQSFPFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holotask::sources

#endif