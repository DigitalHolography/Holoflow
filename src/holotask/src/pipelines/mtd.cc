// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/pipelines/mtd.hh"

#include <algorithm>
#include <cmath>
#include <cwctype>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "holotask/sinks/npy_volume.hh"
#include "holotask/sources/pgm_sequence.hh"

namespace holotask::pipelines {
namespace {
using Settings = std::map<std::string, double>;

Settings read_settings(const std::filesystem::path &directory, const std::string &name) {
  auto path = directory / (name + "_SAV");
  if (!std::filesystem::is_regular_file(path))
    path = directory / name;
  std::ifstream input(path);
  if (!input)
    throw std::invalid_argument("Cannot read MTD settings: " + path.string());
  Settings    result;
  std::string line;
  while (std::getline(input, line)) {
    line = line.substr(0, line.find('#'));
    std::istringstream tokens(line);
    std::string        key;
    double             value;
    if (tokens >> key >> value)
      result[key] = value;
  }
  return result;
}
double setting(const Settings &settings, const std::string &name) {
  const auto it = settings.find(name);
  if (it == settings.end() || !std::isfinite(it->second))
    throw std::invalid_argument("Missing or nonfinite MTD setting: " + name);
  return it->second;
}
int integer_setting(const Settings &settings, const std::string &name) {
  const double value = setting(settings, name);
  if (value < 0 || value > 1000000 || value != std::floor(value))
    throw std::invalid_argument("Invalid integer MTD setting: " + name);
  return static_cast<int>(value);
}
} // namespace

// -------------------------------------------------------------------------------------------------
// Dataset settings
// -------------------------------------------------------------------------------------------------

MTDDatasetSettings load_mtd_dataset(const std::filesystem::path &directory,
                                    std::optional<double>        medium_index) {
  MTDDatasetSettings result;
  result.acquisition_directory   = std::filesystem::absolute(directory);
  const auto camera              = read_settings(directory, "config_manip.txt"),
             recon               = read_settings(directory, "recon.txt");
  result.frame_count             = integer_setting(camera, "NB_HOLO");
  result.recovery.half_size      = integer_setting(camera, "NXMAX");
  result.recovery.center_x       = integer_setting(camera, "CIRCLE_CX");
  result.recovery.center_y       = integer_setting(camera, "CIRCLE_CY");
  result.linear.dimension        = integer_setting(recon, "DIM_FINAL");
  result.linear.hologram_size    = integer_setting(recon, "CCD_ROIX");
  result.linear.wavelength       = setting(camera, "LAMBDA");
  result.linear.background_index = setting(camera, "N0");
  result.medium_index_assumed    = !medium_index.has_value() && !camera.contains("NM");
  result.linear.medium_index =
      medium_index.has_value()
          ? *medium_index
          : (camera.contains("NM") ? setting(camera, "NM") : result.linear.background_index);
  const double tube = setting(camera, "F_TUBE"), objective = setting(camera, "F_OBJ"),
               rf = setting(camera, "RF");
  if (tube <= 0 || objective <= 0 || rf <= 0)
    throw std::invalid_argument("MTD focal lengths and RF must be positive");
  result.linear.pixel_pitch_object = setting(camera, "TPCAM") / ((tube / objective) / rf);
  result.rytov.unwrap_phase        = integer_setting(recon, "DEROUL") != 0;
  if (integer_setting(recon, "BORN") != 0 || integer_setting(recon, "C_ABER") != 1 ||
      integer_setting(recon, "GERCHBERG") != 1)
    throw std::invalid_argument(
        "This MTD pipeline requires Rytov data, aberration correction, and GPS enabled");
  if (result.rytov.unwrap_phase && integer_setting(recon, "VOLKOV") != 1)
    throw std::invalid_argument(
        "MTD settings request a phase-unwrapping method other than the implemented Volkov method");
  result.gps.delta_n_min = static_cast<float>(setting(recon, "DELTA_NMIN"));
  result.gps.delta_n_max = static_cast<float>(setting(recon, "DELTA_NMAX"));
  result.gps.kappa_min   = static_cast<float>(setting(recon, "KAPPA_MIN"));
  result.gps.kappa_max   = static_cast<float>(setting(recon, "KAPPA_MAX"));
  result.gps.iterations  = integer_setting(recon, "NB_ITER_GPS");
  if (result.frame_count < 1)
    throw std::invalid_argument("MTD scan must contain at least one hologram");
  return result;
}

// -------------------------------------------------------------------------------------------------
// Graph construction and registration
// -------------------------------------------------------------------------------------------------

holoflow::core::GraphSpec build_mtd_pipeline(const MTDDatasetSettings &dataset,
                                             const MTDOutputSettings  &output) {
  using namespace holoflow::core;
  if (dataset.frame_count <= 0)
    throw std::invalid_argument("MTD requires a positive scan count");
  std::set<std::filesystem::path> destinations;
  const auto reserve_output = [&destinations](const std::filesystem::path &path,
                                              bool                         sidecar = true) {
    if (path.empty())
      throw std::invalid_argument("MTD output path must not be empty");
    for (const auto &file : {path, std::filesystem::path(path.string() + ".json")}) {
      if (!sidecar && file != path)
        continue;
      auto                  ancestor = std::filesystem::absolute(file).lexically_normal();
      std::filesystem::path suffix;
      while (!std::filesystem::exists(ancestor) && ancestor.has_relative_path()) {
        suffix   = ancestor.filename() / suffix;
        ancestor = ancestor.parent_path();
      }
      auto canonical = (std::filesystem::canonical(ancestor) / suffix).lexically_normal();
#ifdef _WIN32
      auto native = canonical.native();
      std::transform(native.begin(), native.end(), native.begin(),
                     [](wchar_t c) { return std::towlower(c); });
      canonical = native;
#endif
      if (!destinations.insert(canonical).second)
        throw std::invalid_argument("MTD volume, slice, and metadata paths must be distinct");
    }
  };
  if (!output.volume_path.empty())
    reserve_output(output.volume_path);
  if (output.slice_path)
    reserve_output(*output.slice_path);
  if (output.graph_path)
    reserve_output(*output.graph_path, false);
  GraphSpec  graph;
  const auto node = [&graph](const std::string &name, const std::string &kind,
                             nlohmann::json settings) {
    return boost::add_vertex(
        NodeSpec{.name = name, .kind = kind, .settings = std::move(settings), .debug = false},
        graph);
  };
  const auto edge = [&graph](auto source, int source_index, auto target, int target_index) {
    boost::add_edge(source, target, EdgeSpec{source_index, target_index}, graph);
  };
  const auto frames =
      node("mtd_frames", "PgmSequence",
           sources::PgmSequenceSettings{.directory       = dataset.acquisition_directory.string(),
                                        .filename_prefix = "i",
                                        .batch_size      = dataset.frame_count,
                                        .repeat          = false,
                                        .expected_count  = dataset.frame_count});
  const auto mask =
      node("mtd_mask", "PgmSequence",
           sources::PgmSequenceSettings{.directory       = dataset.acquisition_directory.string(),
                                        .filename_prefix = "Image_mask",
                                        .batch_size      = 1,
                                        .repeat          = false,
                                        .expected_count  = 1});
  const auto   field  = node("mtd_field_recovery", "MTDFieldRecovery", dataset.recovery);
  const auto   rytov  = node("mtd_rytov", "RytovCorrection", dataset.rytov);
  const auto   linear = node("mtd_linear3d", "MTDLinear3DReconstruction", dataset.linear);
  const auto   gps    = node("mtd_gps3d", "GPSRefinement3D", dataset.gps);
  const double voxel_pitch =
      dataset.linear.hologram_size * dataset.linear.pixel_pitch_object / dataset.linear.dimension;
  const nlohmann::json metadata = {
      {"acquisition_directory", dataset.acquisition_directory.string()},
      {"frame_count", dataset.frame_count},
      {"recovery", dataset.recovery},
      {"rytov", dataset.rytov},
      {"linear", dataset.linear},
      {"gps", dataset.gps},
      {"voxel_pitch_m", voxel_pitch},
      {"medium_index_assumed", dataset.medium_index_assumed},
      {"phase_unwrapping", "MTD mirrored spectral gradient with local/global damping"}};
  edge(frames, 0, field, 0);
  edge(field, 0, rytov, 0);
  edge(mask, 0, rytov, 1);
  edge(rytov, 0, linear, 0);
  edge(field, 1, linear, 1);
  edge(linear, 0, gps, 0);
  edge(linear, 1, gps, 1);
  if (!output.volume_path.empty()) {
    const auto writer = node("mtd_volume_write", "NpyVolumeWriter",
                             sinks::NpyVolumeSettings{.path      = output.volume_path.string(),
                                                      .metadata  = metadata,
                                                      .overwrite = output.overwrite});
    edge(gps, 0, writer, 0);
  }
  if (output.slice_path.has_value()) {
    const auto slice          = node("mtd_volume_slice", "VolumeSlice", output.slice);
    auto       slice_metadata = metadata;
    slice_metadata["slice"]   = output.slice;
    slice_metadata["axes"] = output.slice.axis == 0 ? "yx" : (output.slice.axis == 1 ? "zx" : "zy");
    const auto slice_writer = node("mtd_slice_write", "NpyVolumeWriter",
                                   sinks::NpyVolumeSettings{.path     = output.slice_path->string(),
                                                            .metadata = slice_metadata,
                                                            .overwrite = output.overwrite});
    edge(gps, 0, slice, 0);
    edge(slice, 0, slice_writer, 0);
  }
  return graph;
}

void register_mtd_tasks(holoflow::core::Registry &registry) {
  if (!registry.is_registered("PgmSequence"))
    registry.register_sync("PgmSequence", std::make_unique<sources::PgmSequenceFactory>());
  registry.register_sync("MTDFieldRecovery", std::make_unique<syncs::MTDFieldRecoveryFactory>());
  registry.register_sync("RytovCorrection", std::make_unique<syncs::RytovCorrectionFactory>());
  registry.register_sync("MTDLinear3DReconstruction",
                         std::make_unique<syncs::MTDLinear3DReconstructionFactory>());
  registry.register_sync("GPSRefinement3D", std::make_unique<syncs::GPSRefinement3DFactory>());
  registry.register_sync("VolumeSlice", std::make_unique<syncs::VolumeSliceFactory>());
  registry.register_sync("NpyVolumeWriter", std::make_unique<sinks::NpyVolumeFactory>());
}
} // namespace holotask::pipelines
