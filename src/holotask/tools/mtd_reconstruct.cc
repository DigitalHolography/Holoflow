// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "holoflow/runtime/compiler.hh"
#include "holotask/pipelines/mtd.hh"
#include <spdlog/spdlog.h>

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
    const auto graph = holotask::pipelines::build_mtd_pipeline(dataset, output);
    if (output.graph_path.has_value()) {
      if (std::filesystem::exists(*output.graph_path) && !output.overwrite)
        throw std::invalid_argument("Graph output already exists; use --overwrite explicitly");
      if (!output.graph_path->parent_path().empty())
        std::filesystem::create_directories(output.graph_path->parent_path());
      std::ofstream file(*output.graph_path);
      file.exceptions(std::ios::badbit | std::ios::failbit);
      file << holoflow::core::to_json(graph).dump(2) << '\n';
    }
    holoflow::core::Registry registry;
    holotask::pipelines::register_mtd_tasks(registry);
    holoflow::runtime::Compiler compiler(registry, {.enable_profiling = false});
    std::cout << "Reconstructing " << dataset.frame_count << " holograms into a "
              << dataset.linear.dimension << " cubed complex volume.\n";
    const auto                   start    = std::chrono::steady_clock::now();
    auto                         compiled = compiler.compile(graph);
    holoflow::runtime::Scheduler scheduler(compiled->graph, compiled->sections,
                                           compiled->resources);
    scheduler.start();
    scheduler.wait();
    if (!std::filesystem::is_regular_file(output.volume_path))
      throw std::runtime_error("Pipeline did not produce the requested volume");
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Wrote " << output.volume_path << " in " << seconds << " seconds.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MTD reconstruction failed: " << error.what() << '\n';
    return 1;
  }
}
