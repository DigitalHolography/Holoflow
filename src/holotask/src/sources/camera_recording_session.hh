#pragma once

#include <cerrno>
#include <cstdio>
#include <mutex>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>

namespace holotask::sources::detail {

class CameraRecordingSession {
public:
  struct Result {
    bool        completed;
    bool        discard_file;
    std::string failure;
  };

  std::stop_token token() const { return stop_.get_token(); }
  void            request_stop() { stop_.request_stop(); }

  // Failure and completion claim the same boundary, so only one can win.
  bool fail(const std::string &message) {
    const std::lock_guard lock(mutex_);
    if (!accepting_failure_ || !failure_.empty() || stop_.stop_requested())
      return false;
    failure_ = message;
    stop_.request_stop();
    return true;
  }

  Result finish(std::string error = {}) {
    const std::lock_guard lock(mutex_);
    accepting_failure_ = false;
    if (!failure_.empty())
      error = failure_;
    const bool completed = !stop_.stop_requested() && error.empty();
    return {completed, !failure_.empty(), std::move(error)};
  }

private:
  std::mutex       mutex_;
  std::stop_source stop_;
  bool             accepting_failure_ = true;
  std::string      failure_;
};

// Called only after the writer has been destroyed, including on Windows where
// an open recording cannot be deleted.
inline std::string remove_incomplete_camera_recording(const std::string &path,
                                                      std::string        failure) {
  if (!path.empty() && std::remove(path.c_str()) != 0 && errno != ENOENT)
    failure += "; failed to remove incomplete recording: " +
               std::error_code(errno, std::generic_category()).message();
  return failure;
}

} // namespace holotask::sources::detail
