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

#include "pipeline/settings.hh"

namespace holovibes::pipeline {
const char *to_string(ImportSource src) {
  switch (src) {
  case ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO:
    return "AmetekS710EuresysCoaxlinkOcto";
  case ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP:
    return "AmetekS711EuresysCoaxlinkQSFP+";
  case ImportSource::AMETEK_S991_EURESYS_COAXLINK_QSFP:
    return "AmetekS991EuresysCoaxlinkQSFP+";
  case ImportSource::HOLOFILE:
    return "Holofile";
  default:
    throw std::runtime_error("unknown import source");
  }
}
} // namespace holovibes::pipeline