// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoflow/runtime/compiler.hh"
#include "holotask/pipelines/mtd.hh"
#include <spdlog/spdlog.h>

namespace {

class OutputTransaction {
public:
  explicit OutputTransaction(bool overwrite) : overwrite_(overwrite) {}
  ~OutputTransaction() {
    for (auto &file : files_) {
      std::error_code error;
      std::filesystem::remove(file.staged, error);
      if (!committed_) {
        if (file.published)
          std::filesystem::remove(file.destination, error);
        if (file.backed_up)
          std::filesystem::rename(file.backup, file.destination, error);
      } else if (file.backed_up) {
        std::filesystem::remove(file.backup, error);
      }
    }
  }
  std::filesystem::path stage(const std::filesystem::path &destination, bool sidecar = false) {
    const auto make_file = [this](std::filesystem::path target) {
      if (std::filesystem::exists(target) && !overwrite_)
        throw std::invalid_argument("MTD output already exists; use --overwrite: " +
                                    target.string());
      auto staged = target;
      staged += ".staged-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                std::to_string(files_.size());
      auto backup = staged;
      backup += ".backup";
      files_.push_back({std::move(target), staged, backup});
      return staged;
    };
    auto staged = make_file(destination);
    if (sidecar) {
      auto &file = files_.back();
      files_.push_back({std::filesystem::path(destination.string() + ".json"),
                        std::filesystem::path(staged.string() + ".json"),
                        std::filesystem::path(file.backup.string() + ".json")});
      if (std::filesystem::exists(files_.back().destination) && !overwrite_)
        throw std::invalid_argument("MTD metadata already exists; use --overwrite: " +
                                    files_.back().destination.string());
    }
    return staged;
  }
  void commit() {
    for (const auto &file : files_) {
      if (!std::filesystem::is_regular_file(file.staged))
        throw std::runtime_error("MTD output was not produced: " + file.staged.string());
    }
    for (auto &file : files_) {
      if (std::filesystem::exists(file.destination)) {
        if (!overwrite_ || !std::filesystem::is_regular_file(file.destination))
          throw std::runtime_error("MTD output cannot be replaced: " + file.destination.string());
        std::filesystem::rename(file.destination, file.backup);
        file.backed_up = true;
      }
    }
    for (auto &file : files_) {
      std::filesystem::rename(file.staged, file.destination);
      file.published = true;
    }
    committed_ = true;
  }

private:
  struct File {
    std::filesystem::path destination, staged, backup;
    bool                  backed_up = false, published = false;
  };
  std::vector<File> files_;
  bool              overwrite_, committed_ = false;
};
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      std::cout << "Usage: holotask_mtd <ACQUIS-directory> <volume.npy> [--medium-index n] "
                   "[--overwrite] [--slice slice.npy] [--z index] [--graph graph.json]\n";
      return argc == 1 ? 0 : 2;
    }
    std::optional<double>                  medium_index;
    holotask::pipelines::MTDOutputSettings output{.volume_path = argv[2]};
    for (int i = 3; i < argc; ++i) {
      const std::string option(argv[i]);
      if (option == "--overwrite")
        output.overwrite = true;
      else {
        if (i + 1 >= argc)
          throw std::invalid_argument("Missing value for " + option);
        const std::string value(argv[++i]);
        if (option == "--medium-index")
          medium_index = std::stod(value);
        else if (option == "--slice")
          output.slice_path = value;
        else if (option == "--z")
          output.slice.index = std::stoi(value);
        else if (option == "--graph")
          output.graph_path = value;
        else
          throw std::invalid_argument("Unknown option: " + option);
      }
    }
    spdlog::set_level(spdlog::level::info);
    const auto dataset = holotask::pipelines::load_mtd_dataset(argv[1], medium_index);
    if (dataset.medium_index_assumed)
      std::cerr << "NM is absent: assuming NM=N0=" << dataset.linear.medium_index
                << ". Use --medium-index to supply the measured value.\n";
    holotask::pipelines::validate_mtd_device_budget(dataset.linear.dimension, false);
    const auto        graph = holotask::pipelines::build_mtd_pipeline(dataset, output);
    OutputTransaction transaction(output.overwrite);
    auto              staged_output = output;
    staged_output.volume_path       = transaction.stage(output.volume_path, true);
    if (output.slice_path)
      staged_output.slice_path = transaction.stage(*output.slice_path, true);
    if (output.graph_path)
      staged_output.graph_path = transaction.stage(*output.graph_path);
    staged_output.overwrite    = false;
    const auto execution_graph = holotask::pipelines::build_mtd_pipeline(dataset, staged_output);
    holoflow::core::Registry registry;
    holotask::pipelines::register_mtd_tasks(registry);
    holoflow::runtime::Compiler compiler(registry, {.enable_profiling = false});
    std::cout << "Reconstructing " << dataset.frame_count << " holograms into a "
              << dataset.linear.dimension << " cubed complex volume.\n";
    const auto                   start    = std::chrono::steady_clock::now();
    auto                         compiled = compiler.compile(execution_graph);
    holoflow::runtime::Scheduler scheduler(compiled->graph, compiled->sections,
                                           compiled->resources);
    scheduler.start();
    scheduler.wait();
    if (staged_output.graph_path) {
      if (!staged_output.graph_path->parent_path().empty())
        std::filesystem::create_directories(staged_output.graph_path->parent_path());
      std::ofstream file(*staged_output.graph_path);
      file.exceptions(std::ios::badbit | std::ios::failbit);
      file << holoflow::core::to_json(graph).dump(2) << '\n';
    }
    transaction.commit();
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Wrote " << output.volume_path << " in " << seconds << " seconds.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MTD reconstruction failed: " << error.what() << '\n';
    return 1;
  }
}
