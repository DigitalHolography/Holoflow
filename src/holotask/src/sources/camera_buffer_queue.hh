#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace holotask::sources::detail {

// One producer, one processing reader, and an optional recording reader.
// A slot remains owned until every subscribed reader has released it.
template <typename Frame> class CameraBufferQueue {
public:
  using DType = Frame;
  enum class PushResult { Accepted, Full, Closed };

  CameraBufferQueue(size_t capacity, std::function<void(const Frame &)> release)
      : slots_(capacity), release_(std::move(release)) {
    if (capacity == 0)
      throw std::invalid_argument("CameraBufferQueue capacity must be > 0");
  }

  ~CameraBufferQueue() {
    // The producer and readers must be joined before destruction.
    for (auto &slot : slots_)
      if (slot.readers != 0)
        release_(slot.frame);
  }

  CameraBufferQueue(const CameraBufferQueue &)            = delete;
  CameraBufferQueue &operator=(const CameraBufferQueue &) = delete;

  [[nodiscard]] PushResult try_push(const Frame &frame) {
    const std::lock_guard lock(mutex_);
    if (closed_)
      return PushResult::Closed;
    auto &slot = slots_[write_ % slots_.size()];
    if (slot.readers != 0)
      return PushResult::Full;
    slot.frame   = frame;
    slot.readers = reader_b_active_ ? 2 : 1;
    ++write_;
    available_.notify_all();
    return PushResult::Accepted;
  }

  const Frame *read_a(const std::atomic<bool> *cancelled = nullptr) {
    return read(read_a_, cancelled);
  }
  const Frame *read_b(const std::atomic<bool> *cancelled = nullptr) {
    return read(read_b_, cancelled);
  }
  void release_a() {
    const std::lock_guard lock(mutex_);
    release(read_a_);
  }
  void release_b() {
    const std::lock_guard lock(mutex_);
    release(read_b_);
  }
  void subscribe_b() {
    const std::lock_guard lock(mutex_);
    if (reader_b_active_)
      throw std::logic_error("CameraBufferQueue reader B is already active");
    read_b_          = write_;
    reader_b_active_ = true;
  }
  void unsubscribe_b() {
    const std::lock_guard lock(mutex_);
    if (!reader_b_active_)
      return;
    reader_b_active_ = false;
    while (read_b_ != write_)
      release(read_b_);
  }
  void close() {
    const std::lock_guard lock(mutex_);
    closed_ = true;
    available_.notify_all();
  }
  size_t capacity() const { return slots_.size(); }
  size_t size() const {
    const std::lock_guard lock(mutex_);
    const auto            oldest = reader_b_active_ && read_b_ < read_a_ ? read_b_ : read_a_;
    return write_ - oldest;
  }
  bool empty() const { return size() == 0; }

private:
  struct Slot {
    Frame    frame{};
    unsigned readers = 0;
  };
  const Frame *read(const size_t &index, const std::atomic<bool> *cancelled) {
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
  void release(size_t &index) {
    if (index == write_)
      throw std::logic_error("CameraBufferQueue release without a frame");
    auto &slot = slots_[index % slots_.size()];
    if (--slot.readers == 0)
      release_(slot.frame);
    ++index;
  }

  mutable std::mutex                 mutex_;
  std::condition_variable            available_;
  std::vector<Slot>                  slots_;
  std::function<void(const Frame &)> release_;
  size_t                             write_ = 0, read_a_ = 0, read_b_ = 0;
  bool                               closed_ = false, reader_b_active_ = false;
};

} // namespace holotask::sources::detail
