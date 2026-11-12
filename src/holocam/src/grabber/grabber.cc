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

#undef max
#include <utility>
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

Grabber::Grabber(uint8_t *buffer, size_t buffer_size, size_t buffer_part_count,
                 Euresys::EGrabberInfo                info_a,
                 std::optional<Euresys::EGrabberInfo> info_b = std::nullopt)
    : grabber_a_{std::make_unique<GrabberType>(info_a)}, grabber_b_{nullptr},
      buffer_part_count_{buffer_part_count} {
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

void Grabber::reset_config() { ctrl_->execute<Euresys::DeviceModule>("DeviceReset"); }

void Grabber::set_offsets(int64_t a, int64_t b) {
  grabber_a_->setInteger<Euresys::StreamModule>("StripeOffset", a);
  assert(grabber_b_ != nullptr);
  grabber_b_->setInteger<Euresys::StreamModule>("StripeOffset", b);
}

std::byte *Grabber::validate_buffer_data(FrameBufferPair &frame) {
  using namespace Euresys;
  constexpr auto DELIVERED = ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS;
  constexpr auto TIMESTAMP = GenTL::BUFFER_INFO_TIMESTAMP;

  auto buffer_a = Buffer(frame.data_a);
  try {
    uint64_t delivered_a = buffer_a.getInfo<uint64_t>(*grabber_a_, DELIVERED);
    if (delivered_a != buffer_part_count_ && log_due(last_rejected_log_)) {
      logger()->error("[Grabber::validate_buffer_data] delivered frame count does not match buffer "
                      "part count: a={}, expected={}",
                      delivered_a, buffer_part_count_);
      return nullptr;
    }

    std::byte *base_a =
        static_cast<std::byte *>(buffer_a.getInfo<void *>(*grabber_a_, GenTL::BUFFER_INFO_BASE));
    if (!grabber_b_)
      return base_a;

    uint64_t   ts_a       = buffer_a.getInfo<uint64_t>(*grabber_a_, TIMESTAMP);
    const auto frame_id_a = buffer_a.getInfo<uint64_t>(*grabber_a_, GenTL::BUFFER_INFO_FRAMEID);

    assert(frame.data_b.has_value());
    auto       buffer_b = Buffer(*frame.data_b);
    std::byte *base_b =
        static_cast<std::byte *>(buffer_b.getInfo<void *>(*grabber_b_, GenTL::BUFFER_INFO_BASE));
    uint64_t   delivered_b = buffer_b.getInfo<uint64_t>(*grabber_b_, DELIVERED);
    uint64_t   ts_b        = buffer_b.getInfo<uint64_t>(*grabber_b_, TIMESTAMP);
    const auto frame_id_b  = buffer_b.getInfo<uint64_t>(*grabber_b_, GenTL::BUFFER_INFO_FRAMEID);

    if (base_a != base_b && log_due(last_rejected_log_)) {
      logger()->error("[Grabber::validate_buffer_data] base pointers differ: a={} b={}", base_a,
                      base_b);
      return nullptr;
    }

    if (delivered_b != buffer_part_count_ && log_due(last_rejected_log_)) {
      logger()->error("[Grabber::validate_buffer_data] delivered frame count does not match buffer "
                      "part count: b={}, expected={}",
                      delivered_a, delivered_b, buffer_part_count_);
      return nullptr;
    }

    if (frame_id_a != frame_id_b && log_due(last_rejected_log_)) {
      if (log_due(last_rejected_log_)) {
        logger()->error(
            "[Grabber::validate_buffer_data] frame ids differ: a={} b={}", frame_id_a, frame_id_b);
      }
      return nullptr;
    }

    const auto ts_delta = ts_a >= ts_b ? ts_a - ts_b : ts_b - ts_a;
    max_ts_delta_since_log_ = std::max(ts_delta, max_ts_delta_since_log_);
    if (log_due(last_accepted_log_)) {
      ++accepted_since_log_;
      logger()->info("[Grabber::validate_buffer_data] accepted {} frames, latest: "
                     "base={}, id={}, max ts delta={} us",
                     accepted_since_log_, static_cast<void *>(base_a), frame_id_a, max_ts_delta_since_log_);
      accepted_since_log_ = 0;
      max_ts_delta_since_log_   = 0;
    }

    return base_a;
  } catch (const Euresys::genapi_error &err) {
    logger()->error("[AmetekS991EuresysCoaxlinkQSFP::acquisition_loop] GenApi error: {}",
                    format_genapi_error(err));
  } catch (const Euresys::gentl_error &err) {
    logger()->error("[AmetekS991EuresysCoaxlinkQSFP::acquisition_loop] GenTL error: {}",
                    err.what());
  } catch (const std::exception &err) {
    logger()->error("[AmetekS991EuresysCoaxlinkQSFP::acquisition_loop] error: {}", err.what());
  }
  return nullptr;
}
} // namespace holocam
#endif