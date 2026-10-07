// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>

namespace spdlog {
class logger;
}

namespace holoflow::runtime {

// Owns snapshots independently of CUDA resources. Only the latest pending report per file is kept.
class DiagnosticsFileWriter {
public:
  using Write = std::function<void(const std::filesystem::path &, const nlohmann::json &)>;
  explicit DiagnosticsFileWriter(Write write = {});
  ~DiagnosticsFileWriter();
  void submit(std::filesystem::path path, nlohmann::json report);
  // For explicit readers/tests after producers stop; never used on the pipeline update path.
  void flush();

private:
  void                                            run();
  Write                                           write_;
  std::shared_ptr<spdlog::logger>                 logger_;
  std::mutex                                      mutex_;
  std::condition_variable                         changed_;
  std::map<std::filesystem::path, nlohmann::json> pending_;
  bool                                            writing_  = false;
  bool                                            stopping_ = false;
  std::thread                                     thread_;
};

// Shared across compilations; resource destruction does not join or flush this worker.
DiagnosticsFileWriter &section_diagnostics_file_writer();

} // namespace holoflow::runtime
