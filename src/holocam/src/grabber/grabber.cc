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

#ifdef HOLOCAM_HAS_EGRABBER
#include "grabber.hh"

#include "logger.hh"

#include <cassert>

namespace holocam {
std::string format_genapi_error(const Euresys::genapi_error &err) {
  std::ostringstream oss;
  oss << "GenApi error: code=" << err.genapi_error_code << ", what=\"" << err.what() << "\"";

  const size_t count = err.parameter_count();
  if (count > 0) {
    oss << ", parameters=[";
    for (size_t i = 0; i < count; ++i) {
      oss << "{";
      switch (err.parameter_type(i)) {
      case GenTL::EuresysCustomGenTL::GENAPI_ERROR_PARAMETER_TYPE_STRING:
        oss << "string:" << err.string_parameter(i);
        break;
      case GenTL::EuresysCustomGenTL::GENAPI_ERROR_PARAMETER_TYPE_INTEGER:
        oss << "int:" << err.integer_parameter(i);
        break;
      case GenTL::EuresysCustomGenTL::GENAPI_ERROR_PARAMETER_TYPE_FLOAT:
        oss << "float:" << err.float_parameter(i);
        break;
      default:
        oss << "unknown";
        break;
      }
      oss << "}";
      if (i + 1 < count) {
        oss << ", ";
      }
    }
    oss << "]";
  }

  return oss.str();
}

Grabber::Grabber(uint8_t *buffer, size_t buffer_size, Euresys::EGrabberInfo info_a,
                 std::optional<Euresys::EGrabberInfo> info_b = std::nullopt)
    : grabber_a_{std::make_unique<GrabberType>(info_a)}, grabber_b_{nullptr} {
  Euresys::UserMemory      memory(buffer, buffer_size);
  Euresys::UserMemoryArray memory_array(memory, buffer_size);

  using Clock = std::chrono::steady_clock;

  const auto start = Clock::now();
  // --------------------------------------------------------------------------
  // Grabber A
  // --------------------------------------------------------------------------
  {
    const auto start = Clock::now();

    const auto range_a = grabber_a_->announceAndQueue(memory_array);

    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    logger()->info("[Grabber::Grabber] grabber A: "
                   "announceAndQueue buffers took {:.3f} ms "
                   "(range {}..{}, {} buffers)",
                   elapsed, range_a.begin, range_a.end, range_a.size());
  }

  // --------------------------------------------------------------------------
  // Grabber B
  // --------------------------------------------------------------------------
  if (info_b.has_value()) {
    grabber_b_ = std::make_unique<GrabberType>(*info_b);

    const auto start = Clock::now();

    const auto range_b = grabber_b_->announceAndQueue(memory_array);

    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    logger()->info("[Grabber::Grabber] grabber B: "
                   "announceAndQueue buffers took {:.3f} ms "
                   "(range {}..{}, {} buffers)",
                   elapsed, range_b.begin, range_b.end, range_b.size());
  }

  const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  logger()->info("[Grabber::Grabber] announcing buffers took {:.3f} ms ", elapsed);
}

void Grabber::requeue(const FrameBufferPair &buffers) noexcept {
  try {
    Euresys::Buffer(buffers.data_a).push(*grabber_a_);
    if (grabber_b_) {
      assert(buffers.data_b.has_value());
      Euresys::Buffer(*buffers.data_b).push(*grabber_b_);
    }
  } catch (const std::exception &e) {
    if (log_due(last_error_log_)) {
      logger()->error("[Grabber::requeue] failed to requeue buffer"
                      "while handling an error: {}",
                      e.what());
    }
  }
}
void Grabber::start_if_not_running() {
  if (!is_running_) {
    if (grabber_b_)
      grabber_b_->start();
    grabber_a_->start();

    logger()->info("[Grabber::start_if_not_running] started");
    is_running_ = true;
  }
}

void Grabber::stop_if_running() {
  if (is_running_) {
    try {
      grabber_a_->stop();
      if (grabber_b_)
        grabber_b_->stop();

      logger()->info("[Grabber::stop_if_running] stopped");
    } catch (const std::exception &e) {
      logger()->error("[Grabber::stop_if_running] failed to stop: {}", e.what());
    }
    is_running_ = false;
  }
}

FrameBufferPair Grabber::pop(uint64_t timeout) {
  auto                                  a = grabber_a_->pop(timeout);
  std::optional<Euresys::NewBufferData> b = std::nullopt;
  if (grabber_b_)
    b = grabber_b_->pop(timeout);

  return {a, b};
}
} // namespace holocam
#endif