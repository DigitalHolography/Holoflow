#include "holovibes/pipeline/worker.hh"

#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <unordered_map>

#include "bug_buster/bug_buster.hh"
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

using json = nlohmann::json;

namespace holovibes::pipeline {

Worker::Worker(dh::TensorDisplayWidget *display_widget, QObject *parent)
    : QObject(parent), display_widget_(display_widget) {
  dh::holovibes_logger()->debug("[Worker::Worker] Pipeline worker initialized");

  // Register the same factories as before.
  compiler_.add_factory("Holofile",
                        std::make_unique<dh::HolofileSourceFactory>());
  compiler_.add_factory(
      "QtDisplay", std::make_unique<dh::QtDisplaySinkFactory>(*display_widget));
  compiler_.add_factory("BatchedSPSC",
                        std::make_unique<dh::BatchedSPSCAccumulatorFactory>());
  compiler_.add_factory(
      "SlidingAverage",
      std::make_unique<dh::SlidingAverageAccumulatorFactory>());
  compiler_.add_factory("Convert", std::make_unique<dh::ConvertTaskFactory>());
  compiler_.add_factory("Identity",
                        std::make_unique<dh::IdentityTaskFactory>());
  compiler_.add_factory("FresnelDiffraction",
                        std::make_unique<dh::FresnelDiffractionTaskFactory>());
  compiler_.add_factory("AngularSpectrum",
                        std::make_unique<dh::AngularSpectrumTaskFactory>());
  compiler_.add_factory("PCA", std::make_unique<dh::PCATaskFactory>());
  compiler_.add_factory("STFT", std::make_unique<dh::STFTTaskFactory>());
  compiler_.add_factory("Average", std::make_unique<dh::AverageTaskFactory>());
  compiler_.add_factory("PercentileClip",
                        std::make_unique<dh::PercentileClipTaskFactory>());
  compiler_.add_factory("FFTShift",
                        std::make_unique<dh::FFTShiftTaskFactory>());
}

void Worker::set_settings(const Settings &settings) { settings_ = settings; }

void Worker::start() {
  if (!settings_) {
    dh::holovibes_logger()->error("[Worker::start] No settings provided");
    emit start_failure();
    return;
  }

  // Build the descriptor graph using settings.
  build_desc_graph();

  // Compile the pipeline.
  model_ = std::nullopt;
  model_ = std::move(compiler_.compile(desc_graph_));
  if (!model_) {
    dh::holovibes_logger()->error(
        "[Worker::start] Pipeline compilation failed");
    emit start_failure();
    return;
  }

  // Create and start the runner.
  runner_ = std::make_unique<holoflow::model::Runner>(*model_);
  runner_->start();

  dh::holovibes_logger()->debug(
      "[Worker::start] Pipeline started successfully");
  emit start_success();
}

void Worker::stop() {
  if (!runner_) {
    dh::holovibes_logger()->error("[Worker::stop] Runner not active");
    emit stop_failure();
    return;
  }

  runner_->stop();
  dh::holovibes_logger()->info("[Worker::stop] Pipeline stopped");
  emit stop_success();
}

void Worker::update() {
  dh::holovibes_logger()->info("[Worker::update] update received");
  if (!runner_) {
    dh::holovibes_logger()->error("[Worker::update] Runner not active");
    emit update_failure();
    return;
  }

  runner_->stop();
  dh::holovibes_logger()->info("[Worker::update] Pipeline stopped");

  if (!settings_) {
    dh::holovibes_logger()->error("[Worker::update] No settings provided");
    emit update_failure();
    return;
  }

  // Build the descriptor graph using settings.
  build_desc_graph();
  dh::holovibes_logger()->info("[Worker::update] descriptor graph built");

  // Compile the pipeline.
  model_ = std::nullopt;
  model_ = std::move(compiler_.compile(desc_graph_));
  if (!model_) {
    dh::holovibes_logger()->error(
        "[Worker::update] Pipeline compilation failed");
    emit update_failure();
    return;
  }

  dh::holovibes_logger()->info("[Worker::update] model compiled");

  // Create and start the runner.
  runner_ = std::make_unique<holoflow::model::Runner>(*model_);
  runner_->start();

  dh::holovibes_logger()->info(
      "[Worker::update] Pipeline started successfully");
  emit update_success();
}

void Worker::build_desc_graph() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  // Reset the descriptor graph and nodes mapping.
  desc_graph_ = holoflow::model::DescriptorGraph();
  nodes_.clear();

  // Build the graph step by step using the helper functions.
  auto source_v = add_source_node();
  auto input_queue_v = add_input_queue_node();
  boost::add_edge(source_v, input_queue_v, desc_graph_);
  auto convert_input_v = add_convert_input_node();
  boost::add_edge(input_queue_v, convert_input_v, desc_graph_);

  auto parent_v = convert_input_v;

  if (s.render_space_transform.has_value()) {
    auto space_v = add_space_transform_node();
    boost::add_edge(parent_v, space_v, desc_graph_);
    parent_v = space_v;
  }

  if (s.render_time_transform.has_value()) {
    auto time_acc_v = add_time_accumulator_node();
    boost::add_edge(parent_v, time_acc_v, desc_graph_);
    auto time_v = add_time_transform_node();
    boost::add_edge(time_acc_v, time_v, desc_graph_);
    parent_v = time_v;
  }

  if (s.render_space_transform.has_value() ||
      s.render_time_transform.has_value()) {
    auto convert_post_v = add_convert_postprocess_node();
    boost::add_edge(parent_v, convert_post_v, desc_graph_);
    parent_v = convert_post_v;
  }

  if (s.render_time_transform.has_value()) {
    auto p_frame_avg_v = add_p_frame_avg_node();
    boost::add_edge(parent_v, p_frame_avg_v, desc_graph_);
    parent_v = p_frame_avg_v;
  }

  if (s.view_fft_shift) {
    auto fft_shift_v = add_fft_shift_node();
    boost::add_edge(parent_v, fft_shift_v, desc_graph_);
    parent_v = fft_shift_v;
  }

  auto img_avg_v = add_image_avg_accumulator_node();
  boost::add_edge(parent_v, img_avg_v, desc_graph_);
  auto percentile_clip_v = add_percentile_clip_node();
  boost::add_edge(img_avg_v, percentile_clip_v, desc_graph_);
  auto convert_output_v = add_convert_output_node();
  boost::add_edge(percentile_clip_v, convert_output_v, desc_graph_);
  auto processed_output_queue_v = add_processed_output_queue_node();
  boost::add_edge(convert_output_v, processed_output_queue_v, desc_graph_);
  auto processed_display_sink_v = add_processed_display_sink_node();
  boost::add_edge(processed_output_queue_v, processed_display_sink_v,
                  desc_graph_);

  // Write DOT file for debugging purposes.
  std::ofstream dot_file("pipeline_graph.dot");
  if (dot_file) {
    boost::write_graphviz(
        dot_file, desc_graph_,
        holoflow::model::DescriptorNodePropertyWriter(desc_graph_),
        holoflow::model::DescriptorEdgePropertyWriter());
    dh::holovibes_logger()->debug(
        "DOT file 'pipeline_graph.dot' created successfully.");
  }
}

holoflow::model::DescriptorVertex Worker::add_node(const std::string &id,
                                                   const std::string &type,
                                                   const json &config) {
  auto v = boost::add_vertex(desc_graph_);
  desc_graph_[v].id = id;
  desc_graph_[v].type = type;
  desc_graph_[v].config = config;
  nodes_[id] = v;
  return v;
}

holoflow::model::DescriptorVertex Worker::add_source_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  json config;
  config["path"] = s.import_file_path;
  config["start_frame"] = s.import_start_index;
  config["end_frame"] = s.import_end_index;
  config["batch_size"] = s.render_batch_size;

  switch (s.import_load_method) {
  case ImportLoadMethod::ReadLive:
    config["load_kind"] = "READ_LIVE";
    break;
  case ImportLoadMethod::LoadInCPU:
    config["load_kind"] = "LOAD_IN_CPU";
    break;
  case ImportLoadMethod::LoadInGPU:
    config["load_kind"] = "LOAD_IN_GPU";
    break;
  }

  return add_node("holofile_source", "Holofile", config);
}

holoflow::model::DescriptorVertex Worker::add_input_queue_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  constexpr size_t TARGET_SIZE = 4096;
  size_t nb_slots = (TARGET_SIZE / s.render_batch_size) * s.render_batch_size;
  json config;
  config["nb_slots"] = nb_slots;
  config["dequeue_batch_size"] = s.render_batch_size;
  return add_node("input_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_convert_input_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  bool skip_to_postprocess = (!s.render_space_transform.has_value() &&
                              !s.render_time_transform.has_value());
  json config;
  config["conversion"] = skip_to_postprocess ? "U8_F32" : "U8_CF32_REAL";
  return add_node("convert_input", "Convert", config);
}

holoflow::model::DescriptorVertex Worker::add_space_transform_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  DH_CHECK(s.render_space_transform.has_value());
  DH_CHECK(s.render_lambda > 0);
  DH_CHECK(s.render_focus > 0);

  constexpr float PIXEL_SIZE = 20e-6f;
  json config;
  config["lambda"] = static_cast<float>(s.render_lambda) * 1e-9f;
  config["z"] = static_cast<float>(s.render_focus) * 1e-3f;
  config["pixel_size"] = PIXEL_SIZE;

  switch (*s.render_space_transform) {
  case RenderSpaceTransform::FresnelDiffraction:
    config["skip_phase_shift"] = true;
    return add_node("fresnel_diffraction", "FresnelDiffraction", config);
  case RenderSpaceTransform::AngularSpectrum:
    return add_node("angular_spectrum_method", "AngularSpectrum", config);
  default:
    DH_BUG("Invalid space transform");
  }
}

holoflow::model::DescriptorVertex Worker::add_time_accumulator_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  size_t max_val = std::max(s.render_time_window, s.render_batch_size);
  size_t target = max_val * 2 + 1;
  size_t gcd_val = std::gcd(s.render_time_window, s.render_batch_size);
  size_t lcm_val = (s.render_time_window / gcd_val) * s.render_batch_size;
  size_t nb_slots = ((target + lcm_val - 1) / lcm_val) * lcm_val;

  json config;
  config["nb_slots"] = nb_slots;
  config["dequeue_batch_size"] = s.render_time_window;
  return add_node("time_accumulator", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_time_transform_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  DH_CHECK(s.render_time_transform);

  json config;
  size_t begin = s.view_p_frame_start;
  size_t end = begin + s.view_p_frame_width;
  switch (*s.render_time_transform) {
  case RenderTimeTransform::PrincipalComponentAnalysis:
    config["begin"] = begin;
    config["end"] = end;
    return add_node("principal_component_analysis", "PCA", config);
  case RenderTimeTransform::ShortTimeFourier:
    return add_node("short_time_fourrier_transform", "STFT", R"({})"_json);
  default:
    DH_BUG("Invalid time transform");
  }
}

holoflow::model::DescriptorVertex Worker::add_convert_postprocess_node() {
  json config;
  config["conversion"] = "CF32_F32_MODU";
  return add_node("convert_postprocess", "Convert", config);
}

holoflow::model::DescriptorVertex Worker::add_p_frame_avg_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  DH_CHECK(s.render_time_transform);

  size_t begin = 0;
  size_t end = 0;
  switch (*s.render_time_transform) {
  case RenderTimeTransform::PrincipalComponentAnalysis:
    begin = 0;
    end = s.view_p_frame_width;
    break;
  case RenderTimeTransform::ShortTimeFourier:
    begin = s.view_p_frame_start;
    end = begin + s.view_p_frame_width;
    break;
  default:
    DH_BUG("Invalid time transform");
  }

  json config;
  config["begin"] = begin;
  config["end"] = end;
  return add_node("p_frame_average", "Average", config);
}

holoflow::model::DescriptorVertex Worker::add_fft_shift_node() {
  json config = json::object();
  return add_node("fft_shift", "FFTShift", config);
}

holoflow::model::DescriptorVertex Worker::add_image_avg_accumulator_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  json config;
  config["window_size"] = s.view_accumulation;
  config["nb_slots"] = s.view_accumulation + 10;
  return add_node("image_avg_accumulator", "SlidingAverage", config);
}

holoflow::model::DescriptorVertex Worker::add_percentile_clip_node() {
  json config;
  config["lower_percentile"] = 0.1f;
  config["upper_percentile"] = 99.9f;
  return add_node("percentile_clip", "PercentileClip", config);
}

holoflow::model::DescriptorVertex Worker::add_convert_output_node() {
  json config;
  config["conversion"] = "F32_U8_SCALED";
  return add_node("convert_output", "Convert", config);
}

holoflow::model::DescriptorVertex Worker::add_processed_output_queue_node() {
  json config;
  config["nb_slots"] = 32;
  config["dequeue_batch_size"] = 1;
  return add_node("processed_output_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_processed_display_sink_node() {
  json config = json::object();
  return add_node("processed_display_sink", "QtDisplay", config);
}

} // namespace holovibes::pipeline
