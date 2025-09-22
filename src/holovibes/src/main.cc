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

#include <QApplication>
#include <cstdlib>

#include "bug.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_exec.hh"
#include "logger.hh"
#include "spdlog/common.h"
#include "tasks/angular_spectrum.hh"
#include "tasks/average.hh"
#include "tasks/batch_queue.hh"
#include "tasks/conversion.hh"
#include "tasks/display_tensor.hh"
#include "tasks/fft_shift.hh"
#include "tasks/fresnel_diffraction.hh"
#include "tasks/holofile.hh"
#include "tasks/memcpy.hh"
#include "tasks/pca.hh"
#include "tasks/pct_clip.hh"
#include "tasks/stft.hh"
#include "ui/tensor_display_widget.hh"

int main() {
  spdlog::set_level(spdlog::level::debug);

  auto path = std::getenv("HOLOVIBES_DEFAULT_SOURCE_PATH");
  HOLOVIBES_CHECK(path != nullptr, "Environment variable HOLOVIBES_DEFAULT_SOURCE_PATH is not set");

  QApplication                       app(__argc, __argv);
  holovibes::ui::TensorDisplayWidget widget;
  widget.show();

  using namespace holovibes::tasks;
  auto angular_spectrum_factory    = std::make_unique<AngularSpectrumFactory>();
  auto average_factory             = std::make_unique<AverageFactory>();
  auto batch_queue_factory         = std::make_unique<BatchQueueFactory>();
  auto conversion_factory          = std::make_unique<ConversionFactory>();
  auto display_factory             = std::make_unique<DisplayTensorFactory>(&widget);
  auto fft_shift_factory           = std::make_unique<FFTShiftFactory>();
  auto fresnel_diffraction_factory = std::make_unique<FresnelDiffractionFactory>();
  auto holofile_factory            = std::make_unique<HolofileFactory>();
  auto memcpy_factory              = std::make_unique<MemcpyFactory>();
  auto pca_factory                 = std::make_unique<PcaFactory>();
  auto pct_clip_factory            = std::make_unique<PctClipFactory>();
  auto stft_factory                = std::make_unique<StftFactory>();
  holoflow::core::Registry registry;
  registry.register_sync("AngularSpectrum", std::move(angular_spectrum_factory));
  registry.register_sync("Average", std::move(average_factory));
  registry.register_async("BatchQueue", std::move(batch_queue_factory));
  registry.register_sync("Conversion", std::move(conversion_factory));
  registry.register_sync("DisplayTensor", std::move(display_factory));
  registry.register_sync("FFTShift", std::move(fft_shift_factory));
  registry.register_sync("FresnelDiffraction", std::move(fresnel_diffraction_factory));
  registry.register_sync("Holofile", std::move(holofile_factory));
  registry.register_sync("Memcpy", std::move(memcpy_factory));
  registry.register_sync("Pca", std::move(pca_factory));
  registry.register_sync("PctClip", std::move(pct_clip_factory));
  registry.register_sync("Stft", std::move(stft_factory));
  holoflow::runtime::Compiler compiler(registry);

  holoflow::core::GraphSpec spec;

  const holoflow::core::NodeSpec holofile_node = {
      .name     = "reader",
      .kind     = "Holofile",
      .settings = nlohmann::json(HolofileSettings{
          .path        = std::string(path),
          .load_kind   = HolofileSettings::LoadKind::Live,
          .start_frame = 0,
          .end_frame   = 65536,
          .batch_size  = 1,
      }),
  };

  auto v_holofile = boost::add_vertex(holofile_node, spec);

  const holoflow::core::NodeSpec in_cpu_queue_node = {
      .name     = "in_cpu_queue",
      .kind     = "BatchQueue",
      .settings = nlohmann::json(BatchQueueSettings{
          .target_capacity = 1024,
          .output_size     = 32,
          .output_stride   = 32,
      }),
  };

  auto v_in_cpu_queue = boost::add_vertex(in_cpu_queue_node, spec);
  boost::add_edge(v_holofile, v_in_cpu_queue, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec cpu_gpu_node = {
      .name     = "cpu_gpu",
      .kind     = "Memcpy",
      .settings = nlohmann::json(MemcpySettings{
          .target = MemcpySettings::Target::Device,
      }),
  };

  auto v_cpu_gpu = boost::add_vertex(cpu_gpu_node, spec);
  boost::add_edge(v_in_cpu_queue, v_cpu_gpu, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec in_gpu_queue_node = {
      .name     = "in_gpu_queue",
      .kind     = "BatchQueue",
      .settings = nlohmann::json(BatchQueueSettings{
          .target_capacity = 1024,
          .output_size     = 32,
          .output_stride   = 32,
      }),
  };

  auto v_in_gpu_queue = boost::add_vertex(in_gpu_queue_node, spec);
  boost::add_edge(v_cpu_gpu, v_in_gpu_queue, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec to_cf32_node = {
      .name     = "to_cf32",
      .kind     = "Conversion",
      .settings = nlohmann::json(ConversionSettings{
          .target   = ConversionSettings::Target::CF32,
          .strategy = ConversionSettings::Strategy::Real,
      }),
  };

  auto v_to_cf32 = boost::add_vertex(to_cf32_node, spec);
  boost::add_edge(v_in_gpu_queue, v_to_cf32, holoflow::core::EdgeSpec{0, 0}, spec);

  // TODO: Add processing nodes here.
  const holoflow::core::NodeSpec fresnel_node = {
      .name     = "fresnel",
      .kind     = "FresnelDiffraction",
      .settings = nlohmann::json(FresnelDiffractionSettings{
          .lambda = 852e-9f,
          .dx     = 20e-6f,
          .dy     = 20e-6f,
          .z      = 380e-3f,
      }),
  };

  auto v_fresnel = boost::add_vertex(fresnel_node, spec);
  boost::add_edge(v_to_cf32, v_fresnel, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec to_f32_node = {
      .name     = "to_f32",
      .kind     = "Conversion",
      .settings = nlohmann::json(ConversionSettings{
          .target   = ConversionSettings::Target::F32,
          .strategy = ConversionSettings::Strategy::Modulus,
      }),
  };

  auto v_to_f32 = boost::add_vertex(to_f32_node, spec);
  boost::add_edge(v_fresnel, v_to_f32, holoflow::core::EdgeSpec{0, 0}, spec);

  // TODO: Add post-processing nodes here.

  const holoflow::core::NodeSpec to_u8_node = {
      .name     = "to_u8",
      .kind     = "Conversion",
      .settings = nlohmann::json(ConversionSettings{
          .target   = ConversionSettings::Target::U8,
          .strategy = ConversionSettings::Strategy::Scaled,
      }),
  };

  auto v_to_u8 = boost::add_vertex(to_u8_node, spec);
  boost::add_edge(v_to_f32, v_to_u8, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec out_gpu_queue_node = {
      .name     = "out_gpu_queue",
      .kind     = "BatchQueue",
      .settings = nlohmann::json(BatchQueueSettings{
          .target_capacity = 1024,
          .output_size     = 32,
          .output_stride   = 32,
      }),
  };

  auto v_out_gpu_queue = boost::add_vertex(out_gpu_queue_node, spec);
  boost::add_edge(v_to_u8, v_out_gpu_queue, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec gpu_cpu_node = {
      .name     = "gpu_cpu",
      .kind     = "Memcpy",
      .settings = nlohmann::json(MemcpySettings{
          .target = MemcpySettings::Target::Host,
      }),
  };

  auto v_gpu_cpu = boost::add_vertex(gpu_cpu_node, spec);
  boost::add_edge(v_out_gpu_queue, v_gpu_cpu, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec out_cpu_queue_node = {
      .name     = "out_cpu_queue",
      .kind     = "BatchQueue",
      .settings = nlohmann::json(BatchQueueSettings{
          .target_capacity = 1024,
          .output_size     = 1,
          .output_stride   = 1,
      }),
  };

  auto v_out_cpu_queue = boost::add_vertex(out_cpu_queue_node, spec);
  boost::add_edge(v_gpu_cpu, v_out_cpu_queue, holoflow::core::EdgeSpec{0, 0}, spec);

  const holoflow::core::NodeSpec display_node = {
      .name     = "display",
      .kind     = "DisplayTensor",
      .settings = nlohmann::json(DisplayTensorSettings{
          .refresh_rate_hz = 60.0f,
      }),
  };

  auto v_display = boost::add_vertex(display_node, spec);
  boost::add_edge(v_out_cpu_queue, v_display, holoflow::core::EdgeSpec{0, 0}, spec);

  auto                         res = compiler.compile(spec);
  holoflow::runtime::Scheduler scheduler(res->graph, res->sections, res->resources);
  scheduler.start();

  return app.exec();
}