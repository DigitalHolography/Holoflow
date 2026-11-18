// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include "pipeline/graph_builder.hh"
#include "pipeline/graph_builder_tasks.hh"
#include "pipeline/settings.hh"

// Graph construction serializes UI sinks too. Keep this inference test independent
// of widget implementations; their settings do not affect the processing cadence.
namespace holovibes::tasks::sinks {
void to_json(nlohmann::json &j, const DisplayTensorSettings &) { j = nlohmann::json::object(); }
void to_json(nlohmann::json &j, const DisplaySignalHistorySettings &) {
  j = nlohmann::json::object();
}
void to_json(nlohmann::json &j, const DisplayZernikeCoefficientsSettings &) {
  j = nlohmann::json::object();
}
} // namespace holovibes::tasks::sinks

namespace holovibes::pipeline {
namespace {

// Only the file reader and UI display are replaced; all processing inference is real.
class SourceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &settings) const override {
    using namespace holoflow::core;
    const auto source = settings.get<holotask::sources::HolofileSettings>();
    const auto location =
        source.load_kind == holotask::sources::HolofileSettings::LoadKind::GPUCached
            ? MemLoc::Device
            : MemLoc::Host;
    return {.input_descs   = {},
            .output_descs  = {TDesc({static_cast<size_t>(source.batch_size), 4, 4}, DType::U8,
                                    location)},
            .in_place      = {},
            .owned_inputs  = {},
            .owned_outputs = {false},
            .kind          = TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return nullptr;
  }
};

class DisplayFactory final : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json &) const override {
    return {.input_descs   = {inputs.begin(), inputs.end()},
            .output_descs  = {},
            .in_place      = {},
            .owned_inputs  = {false},
            .owned_outputs = {},
            .kind          = holoflow::core::TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return nullptr;
  }
};

TEST(GraphBuilderTimingTest, MatchesProcessingStrideAndExportRateForEveryLoadMethod) {
  for (auto method : {LoadMethod::READ_LIVE, LoadMethod::LOAD_IN_CPU, LoadMethod::LOAD_IN_GPU}) {
    for (int stride : {32, 64, 256}) {
      SCOPED_TRACE(static_cast<int>(method));
      SCOPED_TRACE(stride);
      holoflow::core::Registry registry;
      registry.register_sync("Holofile", std::make_unique<SourceFactory>());
      registry.register_sync("DisplayTensorXY", std::make_unique<DisplayFactory>());
      registry.register_sync("Memcpy", std::make_unique<holotask::syncs::MemcpyFactory>());
      registry.register_sync("Conversion", std::make_unique<holotask::syncs::ConversionFactory>());
      registry.register_sync("Reshape", std::make_unique<holonp::ReshapeFactory>());
      registry.register_sync("FFT", std::make_unique<holonp::FFTFactory>());
      registry.register_sync("Slice", std::make_unique<holonp::SliceFactory>());
      registry.register_sync("Copy", std::make_unique<holonp::CopyFactory>());
      registry.register_sync("Concatenate", std::make_unique<holonp::ConcatenateFactory>());
      registry.register_sync("FresnelDiffraction",
                             std::make_unique<holotask::syncs::FresnelDiffractionFactory>());
      registry.register_sync("Mean", std::make_unique<holonp::MeanFactory>());
      registry.register_sync("FfmpegWriter", std::make_unique<holotask::sinks::FfmpegFactory>());
      registry.register_async("BatchQueue",
                              std::make_unique<holotask::asyncs::BatchQueueFactory>());
      registry.register_async("DualReaderBatchQueue",
                              std::make_unique<holotask::asyncs::DualReaderBatchQueueFactory>());
      registry.register_async("SlidingAverage",
                              std::make_unique<holotask::asyncs::SlidingAverageFactory>());

      Settings settings{};
      settings.import_source      = ImportSource::HOLOFILE;
      settings.load_method        = method;
      settings.load_batch         = 16; // Deliberately different from the analysis window.
      settings.time_window        = 32;
      settings.time_stride        = stride;
      settings.time_method        = TimeMethod::FFT;
      settings.time_z_begin       = 1;
      settings.time_z_end         = 2;
      settings.spacial_method     = SpacialMethod::FRESNEL_DIFFRACTION;
      settings.spacial_lambda     = 850e-9f;
      settings.spacial_pixel_size = 20e-6f;
      settings.spacial_z          = 0.1f;
      settings.pp_accumulation    = 1;
      settings.gpu_in_size        = 8; // Smaller than the stride: must still permit progress.
      settings.gpu_out_size       = 8;
      settings.cpu_out_size       = 8;
      settings.input_sampling_frequency_hz = 32000.0;
      settings.recording_method            = RecordingMethod::PROCESSED;
      settings.recording_format            = "matroska";
      settings.recording_codec             = "ffv1";
      settings.recording_path              = "timing.mkv";
      settings.recording_count             = 8;

      const auto graph           = GraphBuilder(settings, registry).build();
      bool       found_window    = false;
      bool       found_recording = false;
      for (auto [vertex, end] = boost::vertices(graph); vertex != end; ++vertex) {
        const auto &node = graph[*vertex];
        if (node.kind == "BatchQueue" && !found_window) {
          found_window = true;
          EXPECT_EQ(node.settings.at("output_size"), 32);
          EXPECT_EQ(node.settings.at("output_stride"), stride);
          EXPECT_GE(node.settings.at("target_capacity").get<int>(), stride);
        }
        if (node.kind == "FfmpegWriter") {
          found_recording = true;
          EXPECT_DOUBLE_EQ(node.settings.at("fps").get<double>(), 32000.0 / stride);
        }
      }
      EXPECT_TRUE(found_window);
      EXPECT_TRUE(found_recording);
    }
  }
}

} // namespace
} // namespace holovibes::pipeline
