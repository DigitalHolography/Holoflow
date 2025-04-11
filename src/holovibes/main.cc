#include <QApplication>
#include <boost/graph/graphviz.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "holoflow/v3/model/compiler.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/runner.hh"
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

using json = nlohmann::json;

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

  QApplication app(argc, argv);
  auto *processed_window = new dh::TensorDisplayWidget(800, 800);
  processed_window->show();

  holoflow::model::ModelCompiler compiler;

  compiler.add_factory("Holofile",
                       std::make_unique<dh::HolofileSourceFactory>());
  compiler.add_factory("QtDisplay", std::make_unique<dh::QtDisplaySinkFactory>(
                                        *processed_window));
  compiler.add_factory("BatchedSPSC",
                       std::make_unique<dh::BatchedSPSCAccumulatorFactory>());
  compiler.add_factory(
      "SlidingAverage",
      std::make_unique<dh::SlidingAverageAccumulatorFactory>());
  compiler.add_factory("Convert", std::make_unique<dh::ConvertTaskFactory>());
  compiler.add_factory("Identity", std::make_unique<dh::IdentityTaskFactory>());
  compiler.add_factory("FresnelDiffraction",
                       std::make_unique<dh::FresnelDiffractionTaskFactory>());
  compiler.add_factory("AngularSpectrum",
                       std::make_unique<dh::AngularSpectrumTaskFactory>());
  compiler.add_factory("PCA", std::make_unique<dh::PCATaskFactory>());
  compiler.add_factory("STFT", std::make_unique<dh::STFTTaskFactory>());
  compiler.add_factory("Average", std::make_unique<dh::AverageTaskFactory>());
  compiler.add_factory("PercentileClip",
                       std::make_unique<dh::PercentileClipTaskFactory>());
  compiler.add_factory("FFTShift", std::make_unique<dh::FFTShiftTaskFactory>());

  holoflow::model::DescriptorGraph graph;

  std::map<std::string, holoflow::model::DescriptorVertex> nodes;

  auto add_node = [&](const std::string &id, const std::string &type,
                      const json &config) {
    holoflow::model::DescriptorVertex v = boost::add_vertex(graph);
    graph[v].id = id;
    graph[v].type = type;
    graph[v].config = config;
    nodes[id] = v;
  };

  // ==========================================================================
  //                     Add Nodes (Pipeline Elements)
  // ==========================================================================

  // Sources
  add_node("holofile_source", "Holofile", R"({
      "path": "D:\\BatchTesting\\250220_GUJ0206_L.holo",
      "start_frame": 0,
      "end_frame": 50000,
      "batch_size": 8,
      "load_kind": "LOAD_IN_CPU"
    })"_json);

  // Sinks
  add_node("processed_widget", "QtDisplay", R"({})"_json);

  // Accumulators
  add_node("input_accumulator", "BatchedSPSC", R"({
      "nb_slots": 1024,
      "dequeue_batch_size": 32
    })"_json);
  add_node("time_accumulator", "BatchedSPSC", R"({
      "nb_slots": 512,
      "dequeue_batch_size": 32
    })"_json);
  add_node("output_accumulator", "BatchedSPSC", R"({
      "nb_slots": 512,
      "dequeue_batch_size": 1
    })"_json);
  add_node("avg_accumulator", "SlidingAverage", R"({
      "nb_slots": 512,
      "window_size": 128
    })"_json);

  // Tasks
  add_node("u8_to_cf32", "Convert", R"({
      "conversion": "U8_CF32_REAL"
    })"_json);
  add_node("cf32_to_f32", "Convert", R"({
      "conversion": "CF32_F32_MODU"
    })"_json);
  add_node("f32_to_u8", "Convert", R"({
      "conversion": "F32_U8_SCALED"
    })"_json);
  add_node("fresnel_diffraction", "FresnelDiffraction", R"({
      "lambda": 852e-9,
      "z": 380e-3,
      "pixel_size": 20e-6,
      "skip_phase_shift": true
    })"_json);
  add_node("pca", "PCA", R"({
      "begin": 0,
      "end": 16
    })"_json);
  add_node("p_frame_avg", "Average", R"({
      "begin": 0,
      "end": 16
    })"_json);
  add_node("percentile_clip", "PercentileClip", R"({
      "lower_percentile": 0.1,
      "upper_percentile": 99.9
    })"_json);
  add_node("fft_shift", "FFTShift", R"({})"_json);

  // ==========================================================================
  //                     Link Nodes (Add Edges)
  // ==========================================================================

  // Helper lambda to add an edge from a source node to a target node, using
  // their IDs.
  auto add_edge_by_id = [&](const std::string &src, const std::string &dst) {
    boost::add_edge(nodes[src], nodes[dst], graph);
  };

  // Create the connections in the order defined by your transaction chaining.
  add_edge_by_id("holofile_source", "input_accumulator");
  add_edge_by_id("input_accumulator", "u8_to_cf32");
  add_edge_by_id("u8_to_cf32", "fresnel_diffraction");
  add_edge_by_id("fresnel_diffraction", "time_accumulator");
  add_edge_by_id("time_accumulator", "pca");
  add_edge_by_id("pca", "cf32_to_f32");
  add_edge_by_id("cf32_to_f32", "p_frame_avg");
  add_edge_by_id("p_frame_avg", "fft_shift");
  add_edge_by_id("fft_shift", "avg_accumulator");
  add_edge_by_id("avg_accumulator", "percentile_clip");
  add_edge_by_id("percentile_clip", "f32_to_u8");
  add_edge_by_id("f32_to_u8", "output_accumulator");
  add_edge_by_id("output_accumulator", "processed_widget");

  // ==========================================================================
  //                     Export Graphviz (DOT) File
  // ==========================================================================

  std::ofstream dotFile("pipeline_graph.dot");
  if (!dotFile) {
    std::cerr << "Error: could not open pipeline_graph.dot for writing."
              << std::endl;
    return 1;
  }

  boost::write_graphviz(dotFile, graph,
                        holoflow::model::DescriptorNodePropertyWriter(graph),
                        holoflow::model::DescriptorEdgePropertyWriter());

  std::cout << "DOT file 'pipeline_graph.dot' created successfully."
            << std::endl;

  auto model = compiler.compile(graph);

  // ==========================================================================
  //                     Run model
  // ==========================================================================

  holoflow::model::Runner runner(model);
  runner.start();
  int result = app.exec();

  dh::holovibes_logger()->info("Stopping the model...");
  runner.stop();
  dh::holovibes_logger()->info("Model stopped.");

  return result;
}