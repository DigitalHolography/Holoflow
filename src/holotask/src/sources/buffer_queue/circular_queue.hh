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

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace holotask::sources {

template <typename T> class CircularQueue {
public:
  using ReleaseCallback = std::function<void(const T &)>;

  explicit CircularQueue(size_t capacity);

  CircularQueue(const CircularQueue &)            = delete;
  CircularQueue &operator=(const CircularQueue &) = delete;
  CircularQueue(CircularQueue &&)                 = delete;
  CircularQueue &operator=(CircularQueue &&)      = delete;

  [[nodiscard]] bool push(const T &value) noexcept;
  [[nodiscard]] bool push(T &&value) noexcept;

  void               pop() noexcept;
  [[nodiscard]] bool peek(T &value) const noexcept;

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool full() const noexcept;

  [[nodiscard]] size_t size() const noexcept;
  [[nodiscard]] size_t capacity() const noexcept;

  void clear() noexcept;

private:
  [[nodiscard]] size_t next(size_t index) const noexcept;

  std::vector<T> buffer_;
  size_t         head_ = 0;
  size_t         tail_ = 0;
  size_t         size_ = 0;
};

} // namespace holotask::sources

#include "circular_queue.hxx"