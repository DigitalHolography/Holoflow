// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "diagnostics_file_writer.hh"

#include <fstream>
#include <spdlog/logger.h>
#include <stdexcept>
#include <utility>

#include "../logger.hh"
#include "holoflow/runtime/tracing.hh"

namespace holoflow::runtime {
namespace {

void write_file(const std::filesystem::path &path, const nlohmann::json &report) {
  std::string text;
  {
    tracing::ScopedTrace format("Format CUDA Graph Diagnostics", "detail");
    text = report.dump(2);
  }
  tracing::ScopedTrace io("Write CUDA Graph Diagnostics File", "detail");
  std::ofstream        file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path);
  file << text;
  file.close();
}

} // namespace

DiagnosticsFileWriter::DiagnosticsFileWriter(Write write)
    : write_(write ? std::move(write) : Write{write_file}), logger_(holoflow::logger()),
      thread_([this] { run(); }) {}

DiagnosticsFileWriter::~DiagnosticsFileWriter() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  changed_.notify_all();
  thread_.join();
}

void DiagnosticsFileWriter::submit(std::filesystem::path path, nlohmann::json report) {
  path = std::filesystem::absolute(path).lexically_normal();
  {
    std::lock_guard lock(mutex_);
    pending_.insert_or_assign(std::move(path), std::move(report));
  }
  changed_.notify_all();
}

void DiagnosticsFileWriter::flush() {
  std::unique_lock lock(mutex_);
  changed_.wait(lock, [this] { return pending_.empty() && !writing_; });
}

void DiagnosticsFileWriter::run() {
  tracing::set_thread_name("CUDA Graph Diagnostics Writer");
  for (;;) {
    std::filesystem::path path;
    nlohmann::json        report;
    {
      std::unique_lock lock(mutex_);
      changed_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
      if (pending_.empty() && stopping_)
        return;
      auto next = pending_.extract(pending_.begin());
      path      = std::move(next.key());
      report    = std::move(next.mapped());
      writing_  = true;
    }
    try {
      write_(path, report);
    } catch (const std::exception &error) {
      try {
        logger_->warn("[CUDA graphs] Could not write {}: {}", path.string(), error.what());
      } catch (...) {
      }
    } catch (...) {
      try {
        logger_->warn("[CUDA graphs] Could not write diagnostics: unknown error");
      } catch (...) {
      }
    }
    {
      std::lock_guard lock(mutex_);
      writing_ = false;
    }
    changed_.notify_all();
  }
}

DiagnosticsFileWriter &section_diagnostics_file_writer() {
  static DiagnosticsFileWriter writer;
  return writer;
}

} // namespace holoflow::runtime
