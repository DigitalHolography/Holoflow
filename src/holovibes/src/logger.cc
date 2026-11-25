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

#include "logger.hh"

#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "holoflow/runtime/tracing.hh"

namespace holovibes {

std::shared_ptr<spdlog::logger> logger() {
  // Keep the worker independent of the registry and alive across updates. One worker preserves
  // queue order; a full queue waits rather than dropping messages. Shutdown drains pending output.
  static auto thread_pool = std::make_shared<spdlog::details::thread_pool>(
      8192, 1, [] { holoflow::runtime::tracing::set_thread_name("Holovibes Logger"); });
  static std::shared_ptr<spdlog::logger> logger = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");

    auto log = std::make_shared<spdlog::async_logger>("holovibes", sink, thread_pool,
                                                      spdlog::async_overflow_policy::block);
    log->set_level(spdlog::default_logger()->level());
    log->flush_on(spdlog::level::warn);

    spdlog::register_logger(log);

    return log;
  }();
  return logger;
}

} // namespace holovibes
