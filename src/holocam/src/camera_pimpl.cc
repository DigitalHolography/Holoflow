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

#include "camera_pimpl.hh"

#include "logger.hh"

#include <cassert>
#include <format>


#ifdef HOLOCAM_HAS_EGRABBER

namespace holocam {

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("error: {}", msg);
    throw std::invalid_argument("error: " + msg);
  }
}

std::optional<Euresys::EGrabberCameraInfo> find_camera(Euresys::EGenTL   &gentl,
                                                       const std::string &model_name) {
  using namespace Euresys;

  EGrabberDiscovery discovery(gentl);
  discovery.discover();

  for (int i = 0; i < discovery.cameraCount(); ++i) {
    auto info = discovery.cameras(i);
    auto g    = EGrabber<>(info);

    try {
      if (g.getString<RemoteModule>("DeviceModelName") == model_name) {
        return info;
      }
    } catch (const Euresys::genapi_error &) {
      try {
        if (g.getString<DeviceModule>("DeviceModelName") == model_name) {
          return info;
        }
      } catch (const Euresys::genapi_error &) {
      }
    }
  }

  return std::nullopt;
}

std::size_t find_grabber_index_for_bank(Euresys::EGrabberCameraInfo &info, std::int64_t bank_id) {
  using namespace Euresys;

  for (std::size_t i = 0; i < info.grabbers.size(); ++i) {
    EGrabber<> g(info.grabbers[i]);
    if (g.getInteger<RemoteModule>("ConnectedBankID") == bank_id) {
      return i;
    }
  }

  throw std::runtime_error(std::format("could not find grabber for ConnectedBankID={}", bank_id));
}
} // namespace

CameraBase::Impl::Impl(const char *model_name, const nlohmann::json &config,
                       size_t expected_grabber_count, size_t buffer_size)
    : config_{config}, gentl_{std::make_unique<Euresys::EGenTL>()},
        buffers_{curaii::make_unique_host_ptr<uint8_t>(buffer_size)}
    {
  auto camera_info = find_camera(*gentl_, model_name);
  check(camera_info.has_value(), std::format("could not find {} camera", model_name));

 
  assert(expected_grabber_count > 0);
  assert(expected_grabber_count <= 2);
  const auto bank_a_index = find_grabber_index_for_bank(*camera_info, 0);
  if(expected_grabber_count == 1) 
        grabber_ = std::make_unique<Grabber>(buffers_.get(), buffer_size, camera_info->grabbers[bank_a_index]);
  else {
    const auto bank_b_index = find_grabber_index_for_bank(*camera_info, 1);
    grabber_ =
        std::make_unique<Grabber>(buffers_.get(), buffer_size, camera_info->grabbers[bank_a_index], camera_info->grabbers[bank_b_index]);
  }
}
} // namespace holocam
#endif