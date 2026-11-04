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

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace holoflow::runtime::tracing {

struct SessionConfig {
  uint32_t buffer_size_kb  = 16 * 1024;
  bool     include_details = true;
};

// One process-wide session. Automatic captures join an existing session without owning it.
// Close measured scopes before stopping a session. Destruction stops without exporting.
class Session {
public:
  [[nodiscard]] static bool available() noexcept;
  [[nodiscard]] static bool active() noexcept;
  // Returns null if another session is active or the SDK is compiled out; throws on setup failure.
  [[nodiscard]] static std::unique_ptr<Session> start(SessionConfig config = {});
  ~Session() noexcept;
  // Stops recording before writing a native binary trace; throws on export failure.
  void stop_and_save(const std::filesystem::path &path);
  Session(const Session &)            = delete;
  Session &operator=(const Session &) = delete;

private:
  class Impl;
  explicit Session(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Best-effort automatic capture. Declared before locks/scopes so export happens after they close.
class Capture {
public:
  explicit Capture(std::filesystem::path path, bool enabled = true) noexcept;
  ~Capture() noexcept;
  Capture(const Capture &)            = delete;
  Capture &operator=(const Capture &) = delete;

private:
  std::filesystem::path    path_;
  std::unique_ptr<Session> session_;
};

enum class Outcome { Completed, Success, Failure };

// Names are copied for NVTX lifetime safety; Perfetto receives them as DynamicString values.
class ScopedTrace {
public:
  explicit ScopedTrace(std::string name, std::string_view category = "lifecycle");
  ~ScopedTrace() noexcept;
  void set_outcome(Outcome outcome) noexcept { outcome_ = outcome; }
  ScopedTrace(const ScopedTrace &)            = delete;
  ScopedTrace &operator=(const ScopedTrace &) = delete;

private:
  std::string name_;
  uint8_t     category_;
  uint64_t    generation_ = 0;
  int         uncaught_exceptions_;
  Outcome     outcome_ = Outcome::Completed;
};

void set_thread_name(std::string_view name) noexcept;

} // namespace holoflow::runtime::tracing
