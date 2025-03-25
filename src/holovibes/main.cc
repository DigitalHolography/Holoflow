#include <glog/logging.h>

#include <QApplication>

#include "holoflow/model.hh"
#include "holoflow/model_descriptor.hh"
#include "holoflow/tensor.hh"
#include "holovibes/accumulators/batched_spsc_accumulator.hh"
#include "holovibes/sinks/qt_display_sink.hh"
#include "holovibes/sources/holofile_source.hh"
#include "holovibes/ui/tensor_display_widget.hh"

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  FLAGS_v = 0;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Welcome to Holovibes!";

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

  // ==========================================================================
  //                     Link nodes
  // ==========================================================================

  descriptor.set_source("holofile_source");

  descriptor.add_child("holofile_source", "input_accumulator");

  descriptor.add_child("input_accumulator", "processed_widget");

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