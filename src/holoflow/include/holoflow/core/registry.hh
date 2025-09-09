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

#include <map>
#include <string>

#include "holoflow/core//tasks.hh"

namespace holoflow::core {

class Registry {
public:
  using Key      = std::string;
  using SyncPtr  = std::unique_ptr<ISyncTaskFactory>;
  using AsyncPtr = std::unique_ptr<IAsyncTaskFactory>;

  /// Register a synchronous task factory.
  /// @throw std::invalid_argument if `kind` is already registered.
  void register_sync(const Key &kind, SyncPtr factory);

  /// Register an asynchronous task factory.
  /// @throw std::invalid_argument if `kind` is already registered.
  void register_async(const Key &kind, AsyncPtr factory);

  /// Lookup a synchronous task factory by kind.
  /// @return A reference to the registered factory.
  /// @throw std::out_of_range if `kind` is not registered.
  const ISyncTaskFactory &get_sync(const Key &kind) const;

  /// Lookup an asynchronous task factory by kind.
  /// @return A reference to the registered factory.
  /// @throw std::out_of_range if `kind` is not registered.
  const IAsyncTaskFactory &get_async(const Key &kind) const;

  /// Lookup any factory (synchronous or asynchronous) by kind.
  /// @return A reference to the registered factory.
  /// @throw std::out_of_range if `kind` is not registered.
  const ITaskFactory &get(const Key &kind) const;

  /// Check whether a synchronous factory with the given key is registered.
  /// @return true if a synchronous factory exists for `kind`
  bool is_sync_registered(const Key &kind) const noexcept;

  /// Check whether an asynchronous factory with the given key is registered.
  /// @return true if an asynchronous factory exists for `kind`.
  bool is_async_registered(const Key &kind) const noexcept;

  /// Check whether any factory (synchronous or asynchronous) with the given key is registered.
  /// @return true if either a synchronous or asynchronous factory exists for `kind`.
  bool is_registered(const Key &kind) const noexcept;

private:
  std::map<Key, SyncPtr>  sync_factories_;  ///< Registered synchronous task factories.
  std::map<Key, AsyncPtr> async_factories_; ///< Registered asynchronous task factories.
};

} // namespace holoflow::core