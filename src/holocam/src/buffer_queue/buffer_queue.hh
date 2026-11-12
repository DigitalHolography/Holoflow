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

#ifdef HOLOCAM_HAS_EGRABBER

#include "grabber/grabber.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <vector>

#include "circular_queue.hh"

namespace holocam {

struct CameraFrame {
  FrameBufferPair       banks;
  std::byte             *base{};
};

class SpinLock {
public:
  void lock();
  void unlock();

private:
  std::atomic<bool> lock_{false};
};

class CameraBufferQueue {
public:
  enum class PushResult { Accepted, Full, Closed };

  using ReadyQueue = CircularQueue<size_t>;
  using FreeQueue  = CircularQueue<size_t>;

  CameraBufferQueue(size_t capacity, std::function<void(const CameraFrame &)> release);
  ~CameraBufferQueue();

  CameraBufferQueue(const CameraBufferQueue &)            = delete;
  CameraBufferQueue &operator=(const CameraBufferQueue &) = delete;

  CameraBufferQueue(CameraBufferQueue &&)            = delete;
  CameraBufferQueue &operator=(CameraBufferQueue &&) = delete;

  void clear();

  [[nodiscard]] PushResult try_push(const CameraFrame &frame);

  const CameraFrame *read_a(const std::atomic<bool> *cancelled = nullptr);
  const CameraFrame *read_b(const std::atomic<bool> *cancelled = nullptr);

  void release_a();
  void release_b();

  void subscribe_b();
  void unsubscribe_b();

  void clear_b();

  void close();
  void open();

  size_t capacity() const { return slots_.size(); }
  size_t size() const;
  bool   empty() const { return size() == 0; }

private:
  struct Slot {
    CameraFrame frame{};
    unsigned    readers = 0;
  };
  const CameraFrame *read(ReadyQueue &queue, const std::atomic<bool> *cancelled);
  void               release(size_t index);

  void fill_free_list();

private:
  mutable SpinLock                         mutex_;
  std::condition_variable_any              available_;
  std::vector<Slot>                        slots_;
  std::function<void(const CameraFrame &)> release_;
  bool                                     closed_ = false, reader_b_active_ = false;

  ReadyQueue ready_queue_a_;
  ReadyQueue ready_queue_b_;
  FreeQueue  free_queue_;
};

} // namespace holocam

#endif // HOLOTASK_HAS_EGRABBER
