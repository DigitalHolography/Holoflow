#pragma once

#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace dh {

inline std::shared_ptr<spdlog::logger> &holovibes_logger() {
  static std::shared_ptr<spdlog::logger> logger = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");

    auto log = std::make_shared<spdlog::logger>("holovibes", sink);
    log->set_level(spdlog::default_logger()->level());
    log->flush_on(spdlog::level::warn);

    spdlog::register_logger(log);

    return log;
  }();
  return logger;
}

} // namespace dh