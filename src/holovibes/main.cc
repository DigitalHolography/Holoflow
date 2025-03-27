#include <QApplication>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "curaii/cufft.hh"
#include "holoflow/model.hh"
#include "holoflow/model_descriptor.hh"
#include "holoflow/tensor.hh"
#include "holovibes/accumulators/batched_spsc_accumulator.hh"
#include "holovibes/sinks/qt_display_sink.hh"
#include "holovibes/sources/holofile_source.hh"
#include "holovibes/tasks/convert_task.hh"
#include "holovibes/tasks/fresnel_diffraction_task.hh"
#include "holovibes/tasks/identity_task.hh"
#include "holovibes/tasks/pca_task.hh"
#include "holovibes/ui/tensor_display_widget.hh"

void setup_global_logger() {
  constexpr std::size_t queue_size = 8192;
  constexpr std::size_t num_threads = 1;
  spdlog::init_thread_pool(queue_size, num_threads);

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  std::vector<spdlog::sink_ptr> sinks{console_sink};

  auto global_logger = std::make_shared<spdlog::logger>(
      "global_logger", sinks.begin(), sinks.end());

  spdlog::set_default_logger(global_logger);
  spdlog::set_level(spdlog::level::debug);
  spdlog::flush_on(spdlog::level::warn);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [thread %t] [%n] [%^%l%$] %v");
}

int main(int argc, char **argv) {
  setup_global_logger();

  // ==========================================================================
  //                     Create display windows
  // ==========================================================================

  QApplication app(argc, argv);

  auto *processed_window = new dh::TensorDisplayWidget(800, 800);
  processed_window->show();

  // ==========================================================================
  //                     Add factories
  // ==========================================================================

  dh::ModelDescriptor descriptor;

  descriptor.add_source_factory("HolofileSourceFactory",
                                std::make_unique<dh::HolofileSourceFactory>());

  descriptor.add_sink_factory(
      "QtDisplaySinkFactory",
      std::make_unique<dh::QtDisplaySinkFactory>(*processed_window));

  descriptor.add_accumulator_factory(
      "BatchedSPSCAccumulatorFactory",
      std::make_unique<dh::BatchedSPSCAccumulatorFactory>());

  descriptor.add_task_factory("ConvertTaskFactory",
                              std::make_unique<dh::ConvertTaskFactory>());

  descriptor.add_task_factory("IdentityTaskFactory",
                              std::make_unique<dh::IdentityTaskFactory>());

  descriptor.add_task_factory(
      "FresnelDiffractionTaskFactory",
      std::make_unique<dh::FresnelDiffractionTaskFactory>());

  descriptor.add_task_factory("PCATaskFactory",
                              std::make_unique<dh::PCATaskFactory>());

  // ==========================================================================
  //                     Add nodes
  // ==========================================================================

  descriptor.add_source("HolofileSourceFactory", "holofile_source",
                        R"({
    "path": "D:\\BatchTesting\\250220_GUJ0206_L.holo",
    "start_frame": 0,
    "end_frame": 10000,
    "batch_size": 32,
    "load_kind": "LOAD_IN_CPU"
  })"_json);

  descriptor.add_sink("QtDisplaySinkFactory", "processed_widget", R"({})"_json);

  descriptor.add_accumulator("BatchedSPSCAccumulatorFactory",
                             "input_accumulator",
                             R"({
      "nb_slots": 1024,
      "dequeue_batch_size": 1
    })"_json);

  descriptor.add_task("ConvertTaskFactory", "convert_input",
                      R"({
      "conversion": "U8_CF32_REAL"
    })"_json);

  descriptor.add_task("ConvertTaskFactory", "convert_postprocess",
                      R"({
      "conversion": "CF32_F32_MODU"
    })"_json);

  descriptor.add_task("ConvertTaskFactory", "convert_output",
                      R"({
      "conversion": "F32_U8_SCALED"
    })"_json);

  descriptor.add_task("FresnelDiffractionTaskFactory", "fresnel_diffraction",
                      R"({
      "lambda": 852e-9,
      "z": 380e-3,
      "pixel_size": 20e-6,
      "skip_phase_shift": true
    })"_json);

  descriptor.add_task("PCATaskFactory", "pca", R"({})"_json);

  // ==========================================================================
  //                     Link nodes
  // ==========================================================================

  descriptor.set_source("holofile_source");
  descriptor.add_child("holofile_source", "input_accumulator");
  descriptor.add_child("input_accumulator", "convert_input");
  descriptor.add_child("convert_input", "fresnel_diffraction");
  descriptor.add_child("fresnel_diffraction", "pca");
  descriptor.add_child("pca", "convert_postprocess");
  descriptor.add_child("convert_postprocess", "convert_output");
  descriptor.add_child("convert_output", "processed_widget");

  // ==========================================================================
  //                     Build model
  // ==========================================================================

  auto model = dh::Model::from_descriptor(descriptor).value();

  // ==========================================================================
  //                     Run model
  // ==========================================================================

  model->start();
  int result = app.exec();
  model->stop();

  return result;
}