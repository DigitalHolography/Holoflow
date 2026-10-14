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

#include "circular_queue.hh"

namespace holotask::sources {

template <typename T> CircularQueue<T>::CircularQueue(size_t capacity) : buffer_(capacity) {
  if (capacity == 0)
    throw std::invalid_argument("CircularQueue capacity must be > 0");
}

template <typename T> [[nodiscard]] bool CircularQueue<T>::push(const T &value) noexcept {
  if (full())
    return false;

  buffer_[tail_] = value;
  tail_          = next(tail_);
  ++size_;

  return true;
}

template <typename T> [[nodiscard]] bool CircularQueue<T>::push(T &&value) noexcept {
  if (full())
    return false;

  buffer_[tail_] = std::move(value);
  tail_          = next(tail_);
  ++size_;

  return true;
}

template <typename T> [[nodiscard]] bool CircularQueue<T>::pop() noexcept {
  if (empty())
    return false;

  head_ = next(head_);
  --size_;

  return true;
}

template <typename T> [[nodiscard]] bool CircularQueue<T>::peek(T &value) const noexcept {
  if (empty())
    return false;

  value = buffer_[head_];

  return true;
}

template <typename T> [[nodiscard]] bool CircularQueue<T>::empty() const noexcept {
  return size_ == 0;
}

template <typename T> [[nodiscard]] bool CircularQueue<T>::full() const noexcept {
  return size_ == buffer_.size();
}

template <typename T> [[nodiscard]] size_t CircularQueue<T>::size() const noexcept { return size_; }

template <typename T> [[nodiscard]] size_t CircularQueue<T>::capacity() const noexcept {
  return buffer_.size();
}

template <typename T> void CircularQueue<T>::clear() noexcept {
  head_ = 0;
  tail_ = 0;
  size_ = 0;
}

template <typename T> [[nodiscard]] size_t CircularQueue<T>::next(size_t index) const noexcept {
  ++index;
  return index == buffer_.size() ? 0 : index;
}

} // namespace holotask::sources