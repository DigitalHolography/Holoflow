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

#include <deque>
#include <mutex>
#include <optional>

namespace holoflow_event {

template <typename T> class BoundedMPMC {
public:
  explicit BoundedMPMC(size_t capacity) : capacity_(capacity) {}

  [[nodiscard]] bool try_push(T &&v) {
    std::scoped_lock lock(mutex_);
    if (queue_.size() >= capacity_) {
      return false;
    }
    queue_.emplace_back(std::move(v));
    return true;
  }

  [[nodiscard]] std::optional<T> try_pop() {
    std::scoped_lock lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    T v = std::move(queue_.front());
    queue_.pop_front();
    return v;
  }

  [[nodiscard]] size_t size() const {
    std::scoped_lock lock(mutex_);
    return queue_.size();
  }

  [[nodiscard]] bool empty() const {
    std::scoped_lock lock(mutex_);
    return queue_.empty();
  }

  [[nodiscard]] size_t capacity() const { return capacity_; }

private:
  size_t             capacity_;
  mutable std::mutex mutex_;
  std::deque<T>      queue_;
};

} // namespace holoflow_event