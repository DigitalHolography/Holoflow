// Copyright 2026 Digital Holography Foundation
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

#include "holocam/camera.hh"

namespace holocam {

class CameraS991 final : public CameraBase {
public:
  explicit CameraS991(const nlohmann::json &config);

  const char *name() const noexcept override { return "AmetekS991EuresysCoaxlinkQSFP+"; }
  const char *short_name() const noexcept override { return "s991"; }
  const char *model_name() const noexcept override { return "Phantom S991"; }
  size_t grabber_count() const noexcept override { return 2; }

protected:
  void configure() override;
};

}