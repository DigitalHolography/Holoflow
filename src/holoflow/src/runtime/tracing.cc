// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "holoflow/runtime/tracing.hh"

#include <atomic>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <nvtx3/nvtx3.hpp>
#include <shared_mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>

#ifdef HOLOFLOW_ENABLE_PERFETTO
#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(perfetto::Category("holoflow.lifecycle"),
                           perfetto::Category("holoflow.compiler"),
                           perfetto::Category("holoflow.scheduler"),
                           perfetto::Category("holoflow.detail"));
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
#endif

namespace holoflow::runtime::tracing {
namespace {
// The shared lock protects emission against stop/start boundaries, not event storage.
std::shared_mutex     session_mutex;
std::mutex            export_mutex;
std::atomic<uint64_t> active_generation{0};
uint64_t              next_generation = 0;

uint8_t category_id(std::string_view category) {
  if (category == "pass")
    return 1;
  if (category == "scheduler")
    return 2;
  if (category == "detail" || category == "syscall")
    return 3;
  return 0;
}

void warn(const char *action, const std::exception &error) noexcept {
  try {
    spdlog::warn("Could not {} Perfetto trace: {}", action, error.what());
  } catch (...) {
  }
}

#ifdef HOLOFLOW_ENABLE_PERFETTO
void initialize() {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    perfetto::TracingInitArgs args;
    args.backends             = perfetto::kInProcessBackend;
    args.log_message_callback = [](perfetto::LogMessageCallbackArgs message) {
      try {
        if (message.level == perfetto::base::kLogError)
          spdlog::error("[Perfetto] {}", message.message);
        else if (message.level == perfetto::base::kLogImportant)
          spdlog::warn("[Perfetto] {}", message.message);
      } catch (...) {
      }
    };
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
    auto track      = perfetto::ProcessTrack::Current();
    auto descriptor = track.Serialize();
    descriptor.mutable_process()->set_process_name("Holoflow");
    perfetto::TrackEvent::SetTrackDescriptor(track, descriptor);
  });
}

bool begin(uint8_t category, const std::string &name) {
#define BEGIN_CATEGORY(cat)                                                                        \
  if (!TRACE_EVENT_CATEGORY_ENABLED(cat))                                                          \
    return false;                                                                                  \
  TRACE_EVENT_BEGIN(cat, perfetto::DynamicString{name});                                           \
  return true
  switch (category) {
  case 1: {
    BEGIN_CATEGORY("holoflow.compiler");
  }
  case 2: {
    BEGIN_CATEGORY("holoflow.scheduler");
  }
  case 3: {
    BEGIN_CATEGORY("holoflow.detail");
  }
  default: {
    BEGIN_CATEGORY("holoflow.lifecycle");
  }
  }
#undef BEGIN_CATEGORY
}

void end(uint8_t category, const char *outcome) {
  switch (category) {
  case 1:
    TRACE_EVENT_END("holoflow.compiler", "outcome", outcome);
    break;
  case 2:
    TRACE_EVENT_END("holoflow.scheduler", "outcome", outcome);
    break;
  case 3:
    TRACE_EVENT_END("holoflow.detail", "outcome", outcome);
    break;
  default:
    TRACE_EVENT_END("holoflow.lifecycle", "outcome", outcome);
    break;
  }
}
#endif
} // namespace

class Session::Impl {
public:
#ifdef HOLOFLOW_ENABLE_PERFETTO
  std::unique_ptr<perfetto::TracingSession> native;
#endif
  uint64_t generation = 0;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool Session::available() noexcept {
#ifdef HOLOFLOW_ENABLE_PERFETTO
  return true;
#else
  return false;
#endif
}

bool Session::active() noexcept { return active_generation.load() != 0; }

std::unique_ptr<Session> Session::start(SessionConfig config) {
#ifdef HOLOFLOW_ENABLE_PERFETTO
  constexpr auto max_buffer_size_kb = (std::numeric_limits<uint32_t>::max)() / 1024;
  if (config.buffer_size_kb == 0 || config.buffer_size_kb > max_buffer_size_kb)
    throw std::invalid_argument("Perfetto buffer size must fit its positive 32-bit byte size");
  std::unique_lock lock(session_mutex);
  if (active())
    return nullptr;
  initialize();
  auto                  impl = std::make_unique<Impl>();
  perfetto::TraceConfig trace_config;
  auto                 *buffer = trace_config.add_buffers();
  buffer->set_size_kb(config.buffer_size_kb);
  buffer->set_fill_policy(perfetto::protos::gen::TraceConfig::BufferConfig::DISCARD);
  auto *source = trace_config.add_data_sources()->mutable_config();
  source->set_name("track_event");
  perfetto::protos::gen::TrackEventConfig categories;
  categories.add_disabled_categories("*");
  for (const auto *category : {"holoflow.lifecycle", "holoflow.compiler", "holoflow.scheduler"})
    categories.add_enabled_categories(category);
  if (config.include_details)
    categories.add_enabled_categories("holoflow.detail");
  source->set_track_event_config_raw(categories.SerializeAsString());
  impl->native = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
  impl->native->Setup(trace_config);
  impl->native->StartBlocking();
  impl->generation = ++next_generation;
  auto result      = std::unique_ptr<Session>(new Session(std::move(impl)));
  active_generation.store(result->impl_->generation);
  return result;
#else
  (void)config;
  return nullptr;
#endif
}

Session::~Session() noexcept {
#ifdef HOLOFLOW_ENABLE_PERFETTO
  try {
    std::unique_lock lock(session_mutex);
    if (impl_->native) {
      active_generation.store(0);
      impl_->native->StopBlocking();
      impl_->native.reset();
    }
  } catch (const std::exception &error) {
    warn("stop", error);
  } catch (...) {
  }
#endif
}

void Session::stop_and_save(const std::filesystem::path &path) {
#ifdef HOLOFLOW_ENABLE_PERFETTO
  std::vector<char> data;
  {
    std::unique_lock lock(session_mutex);
    if (!impl_->native)
      throw std::logic_error("Perfetto session is already stopped");
    active_generation.store(0);
    perfetto::TrackEvent::Flush();
    if (!impl_->native->FlushBlocking()) {
      const std::runtime_error error("flush timed out; capture may be incomplete");
      warn("flush", error);
    }
    impl_->native->StopBlocking();
    try {
      data = impl_->native->ReadTraceBlocking();
    } catch (...) {
      impl_->native.reset();
      throw;
    }
    impl_->native.reset();
  }
  // File writing cannot overlap for automatic captures of the same operation.
  std::lock_guard export_lock(export_mutex);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out.close();
#else
  (void)path;
#endif
}

Capture::Capture(std::filesystem::path path, bool enabled) noexcept : path_(std::move(path)) {
  if (!enabled || path_.empty())
    return;
  try {
    session_ = Session::start();
  } catch (const std::exception &error) {
    warn("start", error);
  } catch (...) {
  }
}

Capture::~Capture() noexcept {
  if (!session_)
    return;
  try {
    session_->stop_and_save(path_);
  } catch (const std::exception &error) {
    warn("save", error);
  } catch (...) {
  }
}

ScopedTrace::ScopedTrace(std::string name, std::string_view category)
    : name_(std::move(name)), category_(category_id(category)),
      uncaught_exceptions_(std::uncaught_exceptions()) {
  nvtxRangePush(name_.c_str());
#ifdef HOLOFLOW_ENABLE_PERFETTO
  if (!Session::active())
    return;
  try {
    std::shared_lock lock(session_mutex);
    const auto       generation = active_generation.load();
    if (generation && begin(category_, name_))
      generation_ = generation;
  } catch (...) {
  }
#endif
}

ScopedTrace::~ScopedTrace() noexcept {
  nvtxRangePop();
#ifdef HOLOFLOW_ENABLE_PERFETTO
  if (!generation_ || active_generation.load() != generation_)
    return;
  try {
    std::shared_lock lock(session_mutex);
    if (active_generation.load() != generation_)
      return;
    const char *outcome = "completed";
    if (outcome_ == Outcome::Success)
      outcome = "success";
    if (outcome_ == Outcome::Failure || std::uncaught_exceptions() > uncaught_exceptions_)
      outcome = "failure";
    end(category_, outcome);
  } catch (...) {
  }
#endif
}

void set_thread_name(std::string_view name) noexcept {
#ifdef HOLOFLOW_ENABLE_PERFETTO
  try {
    // Session initialization owns process metadata; do not initialize the SDK just for naming.
    if (!Session::active())
      return;
    std::shared_lock lock(session_mutex);
    if (!Session::active())
      return;
    auto track      = perfetto::ThreadTrack::Current();
    auto descriptor = track.Serialize();
    descriptor.mutable_thread()->set_thread_name(std::string(name));
    perfetto::TrackEvent::SetTrackDescriptor(track, descriptor);
  } catch (...) {
  }
#else
  (void)name;
#endif
}

} // namespace holoflow::runtime::tracing
