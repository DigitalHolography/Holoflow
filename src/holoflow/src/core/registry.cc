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

#include "holoflow/core/registry.hh"

#include <stdexcept>
#include <utility>

namespace holoflow::core {

void Registry::register_sync(const Key &kind, SyncPtr factory) {
  if (!factory) {
    throw std::invalid_argument("null sync factory");
  }
  if (sync_factories_.find(kind) != sync_factories_.end() ||
      async_factories_.find(kind) != async_factories_.end()) {
    throw std::invalid_argument("factory kind already registered: " + kind);
  }
  sync_factories_.emplace(kind, std::move(factory));
}

void Registry::register_async(const Key &kind, AsyncPtr factory) {
  if (!factory) {
    throw std::invalid_argument("null async factory");
  }
  if (sync_factories_.find(kind) != sync_factories_.end() ||
      async_factories_.find(kind) != async_factories_.end()) {
    throw std::invalid_argument("factory kind already registered: " + kind);
  }
  async_factories_.emplace(kind, std::move(factory));
}

const ISyncTaskFactory &Registry::get_sync(const Key &kind) const {
  auto it = sync_factories_.find(kind);
  if (it == sync_factories_.end()) {
    throw std::out_of_range("unknown sync kind: " + kind);
  }
  return *(it->second);
}

const IAsyncTaskFactory &Registry::get_async(const Key &kind) const {
  auto it = async_factories_.find(kind);
  if (it == async_factories_.end()) {
    throw std::out_of_range("unknown async kind: " + kind);
  }
  return *(it->second);
}

const ITaskFactory &Registry::get(const Key &kind) const {
  if (is_sync_registered(kind)) {
    return get_sync(kind);
  }
  if (is_async_registered(kind)) {
    return get_async(kind);
  }
  throw std::out_of_range("unknown kind: " + kind);
}

bool Registry::is_sync_registered(const Key &kind) const noexcept {
  return sync_factories_.find(kind) != sync_factories_.end();
}

bool Registry::is_async_registered(const Key &kind) const noexcept {
  return async_factories_.find(kind) != async_factories_.end();
}

bool Registry::is_registered(const Key &kind) const noexcept {
  return is_sync_registered(kind) || is_async_registered(kind);
}

} // namespace holoflow::core