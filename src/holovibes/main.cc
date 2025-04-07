#include <QApplication>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "curaii/cufft.hh"
#include "holoflow/model.hh"
#include "holoflow/model_descriptor.hh"
#include "holoflow/tensor.hh"
#include "holoflow/v2/error.hh"
#include "holoflow/v2/model.hh"
#include "holoflow/v2/model_transaction.hh"
#include "holovibes/accumulators/batched_spsc_accumulator.hh"
#include "holovibes/accumulators/sliding_average_accumulator.hh"
#include "holovibes/holovibes.hh"
#include "holovibes/sinks/qt_display_sink.hh"
#include "holovibes/sources/holofile_source.hh"
#include "holovibes/tasks/angular_spectrum_task.hh"
#include "holovibes/tasks/average_task.hh"
#include "holovibes/tasks/convert_task.hh"
#include "holovibes/tasks/fft_shift_task.hh"
#include "holovibes/tasks/fresnel_diffraction_task.hh"
#include "holovibes/tasks/identity_task.hh"
#include "holovibes/tasks/pca_task.hh"
#include "holovibes/tasks/percentile_clip_task.hh"
#include "holovibes/tasks/stft_task.hh"
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
  spdlog::set_level(spdlog::level::trace);
  spdlog::flush_on(spdlog::level::warn);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [thread %t] [%n] [%^%l%$] %v");
}

int main(int argc, char **argv) {
  setup_global_logger();

  auto model_result = dh::v2::Model::create();
  DH_CHECK(model_result);
  auto model = std::move(model_result.value());

  // ==========================================================================
  //                     Create display windows
  // ==========================================================================

  QApplication app(argc, argv);

  auto *processed_window = new dh::TensorDisplayWidget(800, 800);
  processed_window->show();

  // ==========================================================================
  //                     Add factories
  // ==========================================================================

  DH_CHECK(model->register_source_factory(
      "Holofile", std::make_unique<dh::HolofileSourceFactory>()));

  DH_CHECK(model->register_sink_factory(
      "QtDisplay",
      std::make_unique<dh::QtDisplaySinkFactory>(*processed_window)));

  DH_CHECK(model->register_accumulator_factory(
      "BatchedSPSC", std::make_unique<dh::BatchedSPSCAccumulatorFactory>()));

  DH_CHECK(model->register_accumulator_factory(
      "SlidingAverage",
      std::make_unique<dh::SlidingAverageAccumulatorFactory>()));

  DH_CHECK(model->register_task_factory(
      "Convert", std::make_unique<dh::ConvertTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "Identity", std::make_unique<dh::IdentityTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "FresnelDiffraction",
      std::make_unique<dh::FresnelDiffractionTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "AngularSpectrum", std::make_unique<dh::AngularSpectrumTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "PCA", std::make_unique<dh::PCATaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "STFT", std::make_unique<dh::STFTTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "Average", std::make_unique<dh::AverageTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "PercentileClip", std::make_unique<dh::PercentileClipTaskFactory>()));

  DH_CHECK(model->register_task_factory(
      "FFTShift", std::make_unique<dh::FFTShiftTaskFactory>()));

  // ==========================================================================
  //                     Add nodes
  // ==========================================================================

  auto transaction = model->begin_transaction();
  transaction.add_source("holofile_source", "Holofile",
                         R"({
    "path": "D:\\BatchTesting\\250220_GUJ0206_L.holo",
    "start_frame": 0,
    "end_frame": 50000,
    "batch_size": 8,
    "load_kind": "LOAD_IN_CPU"
  })"_json);

  transaction.add_sink("processed_widget", "QtDisplay", R"({})"_json);

  transaction.add_accumulator("input_accumulator", "BatchedSPSC",
                              R"({
      "nb_slots": 1024,
      "dequeue_batch_size": 32
    })"_json);

  transaction.add_accumulator("time_accumulator", "BatchedSPSC",
                              R"({
      "nb_slots": 512,
      "dequeue_batch_size": 32
    })"_json);

  transaction.add_accumulator("output_accumulator", "BatchedSPSC",
                              R"({
      "nb_slots": 512,
      "dequeue_batch_size": 1
    })"_json);

  transaction.add_accumulator("avg_accumulator", "SlidingAverage",
                              R"({
      "nb_slots": 512,
      "window_size": 128
    })"_json);

  transaction.add_task("u8_to_cf32", "Convert",
                       R"({
      "conversion": "U8_CF32_REAL"
    })"_json);

  transaction.add_task("cf32_to_f32", "Convert",
                       R"({
      "conversion": "CF32_F32_MODU"
    })"_json);

  transaction.add_task("f32_to_u8", "Convert",
                       R"({
      "conversion": "F32_U8_SCALED"
    })"_json);

  transaction.add_task("fresnel_diffraction", "FresnelDiffraction",
                       R"({
      "lambda": 852e-9,
      "z": 380e-3,
      "pixel_size": 20e-6,
      "skip_phase_shift": true
    })"_json);

  transaction.add_task("pca", "PCA", R"({
      "begin": 0,
      "end": 16
    })"_json);

  transaction.add_task("p_frame_avg", "Average",
                       R"({
      "begin": 0,
      "end": 16
    })"_json);

  transaction.add_task("percentile_clip", "PercentileClip",
                       R"({
      "lower_percentile": 0.1,
      "upper_percentile": 99.9
    })"_json);

  transaction.add_task("fft_shift", "FFTShift", R"({})"_json);

  // ==========================================================================
  //                     Link nodes
  // ==========================================================================

  transaction.connect("holofile_source", "input_accumulator")
      .connect("input_accumulator", "u8_to_cf32")
      .connect("u8_to_cf32", "fresnel_diffraction")
      .connect("fresnel_diffraction", "time_accumulator")
      .connect("time_accumulator", "pca")
      .connect("pca", "cf32_to_f32")
      .connect("cf32_to_f32", "p_frame_avg")
      .connect("p_frame_avg", "fft_shift")
      .connect("fft_shift", "avg_accumulator")
      .connect("avg_accumulator", "percentile_clip")
      .connect("percentile_clip", "f32_to_u8")
      .connect("f32_to_u8", "output_accumulator")
      .connect("output_accumulator", "processed_widget");

  auto result = transaction.commit();
  if (!result) {
    DH_BUG("Failed to commit transaction: {}", result.error());
  }
}

/*
int main(int argc, char **argv) {
  setup_global_logger();

  // std::vector<dh::v2::Error> errors;
  // errors.emplace_back(dh::v2::Error::make(dh::v2::ErrorType::NotFound,
  //                                         "node '{}' missing", "fft_shift"));
  // errors.emplace_back(
  //     dh::v2::Error::make(dh::v2::ErrorType::ConnectionError,
  //                         "invalid connection from '{}' to '{}'", "A", "B"));

  // auto aggregated = dh::v2::Error::aggregate(
  //     dh::v2::ErrorType::TransactionError, "Model validation failed",
  //     errors);

  // dh::holovibes_logger()->info("{}", aggregated);

  exit(EXIT_FAILURE);

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

  descriptor.add_source_factory("Holofile",
                                std::make_unique<dh::HolofileSourceFactory>());

  descriptor.add_sink_factory(
      "QtDisplay",
      std::make_unique<dh::QtDisplaySinkFactory>(*processed_window));

  descriptor.add_accumulator_factory(
      "BatchedSPSC", std::make_unique<dh::BatchedSPSCAccumulatorFactory>());

  descriptor.add_accumulator_factory(
      "SlidingAverage",
      std::make_unique<dh::SlidingAverageAccumulatorFactory>());

  descriptor.add_task_factory("Convert",
                              std::make_unique<dh::ConvertTaskFactory>());

  descriptor.add_task_factory("Identity",
                              std::make_unique<dh::IdentityTaskFactory>());

  descriptor.add_task_factory(
      "FresnelDiffraction",
      std::make_unique<dh::FresnelDiffractionTaskFactory>());

  descriptor.add_task_factory(
      "AngularSpectrum", std::make_unique<dh::AngularSpectrumTaskFactory>());

  descriptor.add_task_factory("PCA", std::make_unique<dh::PCATaskFactory>());

  descriptor.add_task_factory("STFT", std::make_unique<dh::STFTTaskFactory>());

  descriptor.add_task_factory("Average",
                              std::make_unique<dh::AverageTaskFactory>());

  descriptor.add_task_factory(
      "PercentileClip", std::make_unique<dh::PercentileClipTaskFactory>());

  descriptor.add_task_factory("FFTShift",
                              std::make_unique<dh::FFTShiftTaskFactory>());

  // ==========================================================================
  //                     Add nodes
  // ==========================================================================

  descriptor.add_source("Holofile", "holofile_source",
                        R"({
    "path": "D:\\BatchTesting\\250220_GUJ0206_L.holo",
    "start_frame": 0,
    "end_frame": 50000,
    "batch_size": 8,
    "load_kind": "LOAD_IN_CPU"
  })"_json);

  descriptor.add_sink("QtDisplay", "processed_widget", R"({})"_json);

  descriptor.add_accumulator("BatchedSPSC", "input_accumulator",
                             R"({
      "nb_slots": 1024,
      "dequeue_batch_size": 32
    })"_json);

  descriptor.add_accumulator("BatchedSPSC", "time_accumulator",
                             R"({
      "nb_slots": 512,
      "dequeue_batch_size": 32
    })"_json);

  descriptor.add_accumulator("BatchedSPSC", "output_accumulator",
                             R"({
      "nb_slots": 512,
      "dequeue_batch_size": 1
    })"_json);

  descriptor.add_accumulator("SlidingAverage", "avg_accumulator",
                             R"({
      "nb_slots": 512,
      "window_size": 128
    })"_json);

  descriptor.add_task("Convert", "u8_to_cf32",
                      R"({
      "conversion": "U8_CF32_REAL"
    })"_json);

  descriptor.add_task("Convert", "u16_to_cf32",
                      R"({
      "conversion": "U16_CF32_REAL"
    })"_json);

  descriptor.add_task("Convert", "cf32_to_f32",
                      R"({
      "conversion": "CF32_F32_MODU"
    })"_json);

  descriptor.add_task("Convert", "f32_to_u8",
                      R"({
      "conversion": "F32_U8_SCALED"
    })"_json);

  descriptor.add_task("FresnelDiffraction", "fresnel_diffraction",
                      R"({
      "lambda": 852e-9,
      "z": 380e-3,
      "pixel_size": 20e-6,
      "skip_phase_shift": true
    })"_json);

  descriptor.add_task("AngularSpectrum", "angular_spectrum",
                      R"({
      "lambda": 852e-9,
      "z": 380e-3,
      "pixel_size": 20e-6
    })"_json);

  descriptor.add_task("PCA", "pca", R"({
      "begin": 0,
      "end": 16
    })"_json);

  descriptor.add_task("STFT", "stft", R"({})"_json);

  descriptor.add_task("Average", "p_frame_avg", R"({
      "begin": 0,
      "end": 16
    })"_json);

  descriptor.add_task("Average", "output_average", R"({
      "begin": 0,
      "end": 128
    })"_json);

  descriptor.add_task("PercentileClip", "percentile_clip", R"({})"_json);

  descriptor.add_task("FFTShift", "fft_shift", R"({})"_json);

  // ==========================================================================
  //                     Link nodes
  // ==========================================================================

  descriptor.set_source("holofile_source");
  descriptor.add_child("holofile_source", "input_accumulator");
  descriptor.add_child("input_accumulator", "u8_to_cf32");
  descriptor.add_child("u8_to_cf32", "angular_spectrum");
  descriptor.add_child("angular_spectrum", "time_accumulator");
  descriptor.add_child("time_accumulator", "pca");
  descriptor.add_child("pca", "cf32_to_f32");
  descriptor.add_child("cf32_to_f32", "p_frame_avg");
  descriptor.add_child("p_frame_avg", "fft_shift");
  descriptor.add_child("fft_shift", "avg_accumulator");
  descriptor.add_child("avg_accumulator", "percentile_clip");
  descriptor.add_child("percentile_clip", "f32_to_u8");
  descriptor.add_child("f32_to_u8", "output_accumulator");
  descriptor.add_child("output_accumulator", "processed_widget");

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
}*/