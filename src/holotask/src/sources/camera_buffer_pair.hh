#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <utility>

namespace holotask::sources::detail {

// Acquisition owns popped buffers until a complete frame enters the queue.
// In particular, a timeout on bank B must retain the buffer already read from A.
template <typename BufferData> class CameraBufferPair {
public:
  explicit CameraBufferPair(std::function<void(size_t, const BufferData &)> release)
      : release_(std::move(release)) {}
  ~CameraBufferPair() { release(); }
  CameraBufferPair(const CameraBufferPair &)            = delete;
  CameraBufferPair &operator=(const CameraBufferPair &) = delete;

  template <typename PopA, typename PopB> void poll(PopA pop_a, PopB pop_b) {
    if (!a_)
      a_ = pop_a();
    if (!b_)
      b_ = pop_b();
  }
  const BufferData &a() const { return a_.value(); }
  const BufferData &b() const { return b_.value(); }
  void              transfer() {
    a_.reset();
    b_.reset();
  }
  void release() {
    if (a_) {
      release_(0, *a_);
      a_.reset();
    }
    if (b_) {
      release_(1, *b_);
      b_.reset();
    }
  }

private:
  std::optional<BufferData>                       a_, b_;
  std::function<void(size_t, const BufferData &)> release_;
};

} // namespace holotask::sources::detail
