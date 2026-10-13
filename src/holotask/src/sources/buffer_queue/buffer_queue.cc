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
    : slots_(capacity), release_(std::move(release)) {
  if (capacity == 0)
    throw std::invalid_argument("CameraBufferQueue capacity must be > 0");
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

  write_  = 0;
  read_a_ = 0;
  read_b_ = 0;
  available_.notify_all();
}

[[nodiscard]] CameraBufferQueue::PushResult CameraBufferQueue::try_push(const CameraFrame &frame) {
  {
    const std::lock_guard lock(mutex_);
    if (closed_)
      return PushResult::Closed;
    auto &slot = slots_[write_ % slots_.size()];
    if (slot.readers != 0)
      return PushResult::Full;
    slot.frame   = frame;
    slot.readers = reader_b_active_ ? 2 : 1;
    ++write_;
  }
  available_.notify_all();
  return PushResult::Accepted;
}

const CameraFrame *CameraBufferQueue::read_a(const std::atomic<bool> *cancelled) {
  return read(read_a_, cancelled);
}
const CameraFrame *CameraBufferQueue::read_b(const std::atomic<bool> *cancelled) {
  return read(read_b_, cancelled);
}

void CameraBufferQueue::release_a() {
  const std::lock_guard lock(mutex_);
  release(read_a_);
}
void CameraBufferQueue::release_b() {
  const std::lock_guard lock(mutex_);
  release(read_b_);
}
void CameraBufferQueue::subscribe_b() {
  const std::lock_guard lock(mutex_);
  if (reader_b_active_)
    throw std::logic_error("CameraBufferQueue reader B is already active");
  read_b_          = write_;
  reader_b_active_ = true;
}
void CameraBufferQueue::unsubscribe_b() {
  const std::lock_guard lock(mutex_);
  if (!reader_b_active_)
    return;
  reader_b_active_ = false;
  while (read_b_ != write_)
    release(read_b_);
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
  const auto            oldest = reader_b_active_ && read_b_ < read_a_ ? read_b_ : read_a_;
  return write_ - oldest;
}

const CameraFrame *CameraBufferQueue::read(const size_t            &index,
                                           const std::atomic<bool> *cancelled) {
  std::unique_lock lock(mutex_);
  // Timed waits also observe external pipeline cancellation without requiring
  // its owner to notify this queue.
  while (!closed_ && !(cancelled && cancelled->load(std::memory_order_acquire))) {
    if (index != write_)
      return &slots_[index % slots_.size()].frame;
    available_.wait_for(lock, std::chrono::milliseconds(10));
  }
  return nullptr;
}
void CameraBufferQueue::release(size_t &index) {
  if (index == write_)
    throw std::logic_error("CameraBufferQueue release without a frame");
  auto &slot = slots_[index % slots_.size()];
  if (--slot.readers == 0)
    release_(slot.frame);
  ++index;
}

} // namespace holotask::sources