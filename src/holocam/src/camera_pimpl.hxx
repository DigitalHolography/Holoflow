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

#ifdef HOLOCAM_HAS_EGRABBER

namespace holocam {

template <typename Module, typename T>
void CameraBase::Impl::set_required(const char *name, const T &value) {
  grabber_->set_required(name, value);
}

template <typename Module, typename T>
void CameraBase::Impl::set_opt(const char *name, const T &value) {
  grabber_->set_opt(name, value);
}

} // namespace holocam
#endif