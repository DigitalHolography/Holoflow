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
#include <variant>

namespace spdlog {
class logger;
}

namespace holoflow::runtime {

// Owns snapshots independently of CUDA resources. Only the latest pending report per file is kept.
class DiagnosticsFileWriter {
public:
  using Render = std::function<std::string()>;
  using Write  = std::function<void(const std::filesystem::path &, const nlohmann::json &)>;
  explicit DiagnosticsFileWriter(Write write = {});
  ~DiagnosticsFileWriter();
  void submit(std::filesystem::path path, nlohmann::json report);
  // Render must own its inputs; it runs on the writer, never on the submitting thread.
  void submit_text(std::filesystem::path path, Render render);
  // For explicit readers/tests after producers stop; never used on the pipeline update path.
  void flush();

private:
  using Job = std::variant<nlohmann::json, Render>;
  void                                 enqueue(std::filesystem::path path, Job job);
  void                                 run();
  Write                                write_;
  std::shared_ptr<spdlog::logger>      logger_;
  std::mutex                           mutex_;
  std::condition_variable              changed_;
  std::map<std::filesystem::path, Job> pending_;
  bool                                 writing_  = false;
  bool                                 stopping_ = false;
  std::thread                          thread_;
};

// Shared across compilations; resource destruction does not join or flush this worker.
DiagnosticsFileWriter &section_diagnostics_file_writer();

} // namespace holoflow::runtime
