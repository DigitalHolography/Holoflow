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

#include "buffer_queue.hh"

#include <cassert>
#undef min
#undef max
#include <limits>
namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// CameraBufferPair
// -------------------------------------------------------------------------------------------------

CameraBufferPair::CameraBufferPair(Grabber &grabber_a, Grabber &grabber_b)
    : grabber_a_{grabber_a}, grabber_b_{grabber_b} {}

CameraBufferPair::~CameraBufferPair() { release(); }

void CameraBufferPair::poll(uint64_t timeout) {
  if (!a_)
    a_ = grabber_a_.pop(timeout);
  if (!b_)
    b_ = grabber_b_.pop(timeout);
}

void CameraBufferPair::transfer() {
  a_.reset();
  b_.reset();
}

void CameraBufferPair::release() {
  if (a_) {
    grabber_a_.requeue_buffer(*a_);
    a_.reset();
  }
  if (b_) {
    grabber_b_.requeue_buffer(*b_);
    b_.reset();
  }
}

// -------------------------------------------------------------------------------------------------
// SpinLock
// -------------------------------------------------------------------------------------------------

void SpinLock::lock() {
  while (lock_.exchange(true))
    ;
}

void SpinLock::unlock() { lock_.store(false, std::memory_order_release); }

// -------------------------------------------------------------------------------------------------
// CameraBufferQueue
// -------------------------------------------------------------------------------------------------

CameraBufferQueue::CameraBufferQueue(size_t                                   capacity,
                                     std::function<void(const CameraFrame &)> release)
    : slots_(capacity), release_(std::move(release)), ready_queue_a_{capacity},
      ready_queue_b_{capacity}, free_queue_{capacity} {
  if (capacity == 0)
    throw std::invalid_argument("CameraBufferQueue capacity must be > 0");

  fill_free_list();
}

void CameraBufferQueue::fill_free_list() {
  for (size_t i = 0; i < slots_.size(); ++i) {
    auto pushed = free_queue_.push(i);
    assert(pushed);
    (void)pushed; // for release build to avoid unused variable warning
  }
}

CameraBufferQueue::~CameraBufferQueue() {
  // The producer and readers must be joined before destruction.
  clear();
}

void CameraBufferQueue::clear() {
  const std::lock_guard lock(mutex_);
  for (auto &slot : slots_)
    if (slot.readers != 0) {
      release_(slot.frame);
      slot.readers = 0;
    }

  ready_queue_a_.clear();
  ready_queue_b_.clear();
  free_queue_.clear();

  fill_free_list();

  available_.notify_all();
}

[[nodiscard]] CameraBufferQueue::PushResult CameraBufferQueue::try_push(const CameraFrame &frame) {
  {
    const std::lock_guard lock(mutex_);
    if (closed_)
      return PushResult::Closed;

    size_t write_index = std::numeric_limits<size_t>::max();
    if (!free_queue_.peek(write_index))
      return PushResult::Full;

    free_queue_.pop();

    auto &slot = slots_[write_index];
    assert(slot.readers == 0);

    slot.frame          = frame;
    const bool pushed_a = ready_queue_a_.push(write_index);
    assert(pushed_a);
    (void)pushed_a;

    if (reader_b_active_) {
      const bool pushed_b = ready_queue_b_.push(write_index);
      assert(pushed_b);
      (void)pushed_b;
    }

    slot.readers = reader_b_active_ ? 2u : 1u;
  }

  available_.notify_all();
  return PushResult::Accepted;
}

const CameraFrame *CameraBufferQueue::read_a(const std::atomic<bool> *cancelled) {
  return read(ready_queue_a_, cancelled);
}
const CameraFrame *CameraBufferQueue::read_b(const std::atomic<bool> *cancelled) {
  return read(ready_queue_b_, cancelled);
}

void CameraBufferQueue::release_a() {
  const std::lock_guard lock(mutex_);
  auto                  index = std::numeric_limits<size_t>::max();
  while (!ready_queue_a_.peek(index))
    ;
  release(index);
  ready_queue_a_.pop();
}

void CameraBufferQueue::release_b() {
  const std::lock_guard lock(mutex_);
  auto                  index = std::numeric_limits<size_t>::max();
  while (!ready_queue_b_.peek(index))
    ;

  release(index);
  ready_queue_b_.pop();
}

void CameraBufferQueue::subscribe_b() {
  const std::lock_guard lock(mutex_);
  if (reader_b_active_)
    throw std::logic_error("CameraBufferQueue reader B is already active");
  reader_b_active_ = true;
}

void CameraBufferQueue::unsubscribe_b() {
  const std::lock_guard lock(mutex_);
  if (!reader_b_active_)
    return;
  reader_b_active_ = false;
}

void CameraBufferQueue::clear_b() {
  std::lock_guard lock(mutex_);
  size_t          index = std::numeric_limits<size_t>::max();
  while (ready_queue_b_.peek(index)) {
    release(index);
    ready_queue_b_.pop();
  }
}

void CameraBufferQueue::close() {
  {
    const std::lock_guard lock(mutex_);
    closed_ = true;
  }
  available_.notify_all();
}

size_t CameraBufferQueue::size() const {
  const std::lock_guard lock(mutex_);
  return slots_.size() - free_queue_.size();
}

const CameraFrame *CameraBufferQueue::read(ReadyQueue &queue, const std::atomic<bool> *cancelled) {
  std::unique_lock lock(mutex_);
  // Timed waits also observe external pipeline cancellation without requiring
  // its owner to notify this queue.
  while (!closed_ && !(cancelled && cancelled->load(std::memory_order_acquire))) {
    size_t index = std::numeric_limits<size_t>::max();
    if (queue.peek(index))
      return &slots_[index].frame;
    available_.wait_for(lock, std::chrono::milliseconds(10));
  }
  return nullptr;
}

void CameraBufferQueue::release(size_t index) {
  auto &slot = slots_[index];
  if (--slot.readers == 0) {
    release_(slot.frame);
    const bool pushed = free_queue_.push(index);
    assert(pushed);
    (void) pushed;
  }
}

} // namespace holotask::sources