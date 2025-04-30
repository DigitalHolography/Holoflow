#include "holovibes/pipeline/worker.hh"

#include <boost/graph/graphviz.hpp>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <unordered_map>

#include "bug_buster/bug_buster.hh"
#include "holovibes/accumulators/batched_spsc_accumulator.hh"
#include "holovibes/accumulators/gate_accumulator.hh"
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
#include "holovibes/tasks/memcpy_task.hh"
#include "holovibes/tasks/pca_task.hh"
#include "holovibes/tasks/percentile_clip_task.hh"
#include "holovibes/tasks/stft_task.hh"

using json = nlohmann::json;

namespace holovibes::pipeline {

size_t ceil_lcm(size_t x, size_t y, size_t k) {
  auto base = std::lcm(x, y);
  DH_CHECK(base != 0);
  auto n = (k + base - 1) / base;
  return n * base;
}

Worker::Worker(dh::TensorDisplayWidget *processed_display_widget,
               dh::TensorDisplayWidget *raw_record_display_widget,
               QObject *parent)
    : QObject(parent), processed_display_widget_(processed_display_widget),
      raw_record_display_widget_(raw_record_display_widget) {
  dh::holovibes_logger()->debug("[Worker::Worker] Pipeline worker initialized");

  // clang-format off
  compiler_.add_factory<dh::HolofileSourceFactory>("Holofile");
  compiler_.add_factory<dh::QtDisplaySinkFactory>("QtDisplay", *processed_display_widget_);
  compiler_.add_factory<dh::QtDisplaySinkFactory>("QtDisplayRawRecord", *raw_record_display_widget_);
  compiler_.add_factory<accumulators::BatchedSPSCFactory>("BatchedSPSC");
  compiler_.add_factory<dh::SlidingAverageAccumulatorFactory>("SlidingAverage");
  compiler_.add_factory<holovibes::accumulators::GateFactory>("Gate");
  compiler_.add_factory<dh::ConvertTaskFactory>("Convert");
  compiler_.add_factory<dh::IdentityTaskFactory>("Identity");
  compiler_.add_factory<dh::FresnelDiffractionTaskFactory>("FresnelDiffraction");
  compiler_.add_factory<dh::AngularSpectrumTaskFactory>("AngularSpectrum");
  compiler_.add_factory<dh::PCATaskFactory>("PCA");
  compiler_.add_factory<dh::STFTTaskFactory>("STFT");
  compiler_.add_factory<dh::AverageTaskFactory>("Average");
  compiler_.add_factory<dh::PercentileClipTaskFactory>("PercentileClip");
  compiler_.add_factory<dh::FFTShiftTaskFactory>("FFTShift");
  compiler_.add_factory<dh::MemcpyTaskFactory>("Memcpy");
  // clang-format on
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
  try {
    model_ = std::move(compiler_.compile(desc_graph_, event_listeners_));
  } catch (std::invalid_argument &e) {
    dh::holovibes_logger()->error(
        "[Worker::start] Model compilation failed: {}", e.what());
    emit start_failure();
    return;
  } catch (std::exception &e) {
    dh::holovibes_logger()->error(
        "[Worker::start] Model compilation failed: {}", e.what());
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
  try {
    model_ = std::move(compiler_.compile(desc_graph_, event_listeners_));
  } catch (std::invalid_argument &e) {
    dh::holovibes_logger()->error(
        "[Worker::update] Model compilation failed: {}", e.what());
    emit update_failure();
    return;
  } catch (std::exception &e) {
    dh::holovibes_logger()->error(
        "[Worker::update] Model compilation failed: {}", e.what());
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

void Worker::start_export() {
  dh::holovibes_logger()->info("[Worker::start_export] starting export");
  json event;
  event["action"] = "start";
  try {
    runner_->send_event("raw_record_gate", event);
  } catch (std::exception e) {
    dh::holovibes_logger()->warn("[Worker::start_export] error: {}", e.what());
    emit start_export_failure();
    return;
  }
  emit start_export_success();
}

void Worker::stop_export() {
  dh::holovibes_logger()->info("[Worker::stop_export] stoping export");
  json event;
  event["action"] = "stop";
  try {
    runner_->send_event("raw_record_gate", event);
  } catch (std::exception e) {
    dh::holovibes_logger()->warn("[Worker::stop_export] error: {}", e.what());
    emit stop_export_failure();
    return;
  }

  emit stop_export_success();
}

void Worker::build_desc_graph() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  if (s.render_time_stride % s.render_time_window == 0) {
    opti_early_stride_ = true;
  }

  // Reset the descriptor graph and nodes mapping.
  desc_graph_ = holoflow::model::DescriptorGraph();
  event_listeners_.clear();
  nodes_.clear();

  // Build the graph step by step using the helper functions.
  auto source_v = add_source_node();
  auto parent_v = source_v;

  auto raw_record_gate_v = add_raw_record_gate_node();
  boost::add_edge(source_v, raw_record_gate_v, desc_graph_);
  auto identity_v = add_raw_identity_node();
  boost::add_edge(raw_record_gate_v, identity_v, desc_graph_);
  auto raw_record_acc_v = add_raw_record_accumulator_node();
  boost::add_edge(identity_v, raw_record_acc_v, desc_graph_);
  auto raw_record_display_v = add_raw_record_display_sink_node();
  boost::add_edge(raw_record_acc_v, raw_record_display_v, desc_graph_);

  if (s.import_load_method != ImportLoadMethod::LoadInGPU) {
    // Output is on CPU memory, need to send it to GPU.
    // We use CPU queue to enable parralelisme with
    // record and optimize stride.
    auto identity_v = add_node("identity_0", "Identity", json::object());
    boost::add_edge(parent_v, identity_v, desc_graph_);
    auto cpu_queue_v = add_cpu_input_queue_node();
    boost::add_edge(identity_v, cpu_queue_v, desc_graph_);
    auto memcpy_v = add_cpy_cpu_to_gpu_node();
    boost::add_edge(cpu_queue_v, memcpy_v, desc_graph_);
    parent_v = memcpy_v;
  }

  auto input_queue_v = add_input_queue_node();
  boost::add_edge(parent_v, input_queue_v, desc_graph_);
  auto convert_input_v = add_convert_input_node();
  boost::add_edge(input_queue_v, convert_input_v, desc_graph_);
  parent_v = convert_input_v;

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

  if (!s.render_time_transform && s.render_batch_size != 1) {
    auto split_v = add_split_axis_0_node();
    boost::add_edge(parent_v, split_v, desc_graph_);
    auto identity_v = add_identity_node();
    boost::add_edge(split_v, identity_v, desc_graph_);
    parent_v = identity_v;
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

holoflow::model::DescriptorVertex Worker::add_raw_record_gate_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;

  event_listeners_["raw_record_gate"].push_back(
      [this](const nlohmann::json &evt) {
        auto action = evt.value("action", "");
        if (action == "started")
          emit start_export_success();
        else if (action == "start_failed")
          emit start_export_failure();
        else if (action == "stopped")
          emit stop_export_success();
        else if (action == "stop_failed")
          emit stop_export_failure();
      });

  accumulators::GateParams config;
  config.is_on = s.export_activated;
  config.target = s.export_frame_count;
  return add_node("raw_record_gate", "Gate", config);
}

holoflow::model::DescriptorVertex Worker::add_raw_identity_node() {
  json config = json::object();
  return add_node("raw_identity", "Identity", config);
}

holoflow::model::DescriptorVertex Worker::add_raw_record_accumulator_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  auto batch_sz = s.render_batch_size;
  auto export_count = s.export_frame_count.value_or(4096);

  size_t cap = export_count + batch_sz;

  accumulators::BatchedSPSCParams config;
  config.nb_slots = ceil_lcm(batch_sz, 1, cap);
  config.dequeue_batch_size = 1;
  return add_node("raw_record_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_raw_record_display_sink_node() {
  json config = json::object();
  return add_node("raw_record_display_sink", "QtDisplayRawRecord", config);
}

holoflow::model::DescriptorVertex Worker::add_cpu_input_queue_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  DH_CHECK(s.import_load_method != ImportLoadMethod::LoadInGPU);
  auto batch_sz = s.render_batch_size;
  auto stride = s.render_time_stride;

  size_t cap = 4096 + batch_sz;

  accumulators::BatchedSPSCParams config;
  config.nb_slots = ceil_lcm(batch_sz, batch_sz, cap);
  config.dequeue_batch_size = batch_sz;

  if (opti_early_stride_) {
    config.nb_slots = ceil_lcm(batch_sz, stride, cap);
    config.stride = stride;
  }

  return add_node("cpu_input_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_cpy_cpu_to_gpu_node() {
  json config;
  config["kind"] = "HOST_TO_DEVICE";
  return add_node("memcpy_cpu_to_gpu", "Memcpy", config);
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
  auto batch_sz = s.render_batch_size;
  auto stride = s.render_time_stride;

  size_t cap = 4096 + batch_sz;

  accumulators::BatchedSPSCParams config;
  config.nb_slots = ceil_lcm(batch_sz, batch_sz, cap);
  config.dequeue_batch_size = batch_sz;

  if (opti_early_stride_ &&
      s.import_load_method == ImportLoadMethod::LoadInGPU) {
    config.nb_slots = ceil_lcm(batch_sz, stride, cap);
    config.stride = stride;
  }

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
  auto time_win = s.render_time_window;
  auto batch_sz = s.render_batch_size;
  auto stride = s.render_time_stride;

  size_t cap = std::max(stride, batch_sz);
  cap = cap * 2 + batch_sz;

  accumulators::BatchedSPSCParams config;
  config.nb_slots = ceil_lcm(batch_sz, time_win, cap);
  config.dequeue_batch_size = time_win;

  if (!opti_early_stride_) {
    config.nb_slots = ceil_lcm(batch_sz, stride, cap);
    config.stride = stride;
  }

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

holoflow::model::DescriptorVertex Worker::add_split_axis_0_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  DH_CHECK(!s.render_time_transform);
  DH_CHECK(s.render_batch_size != 1);
  auto batch_sz = s.render_batch_size;

  size_t cap = batch_sz * 3;

  accumulators::BatchedSPSCParams config;
  config.nb_slots = ceil_lcm(batch_sz, 1, cap);
  config.dequeue_batch_size = 1;
  return add_node("split_axis_0", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_identity_node() {
  DH_CHECK(settings_);
  const auto &s = *settings_;
  DH_CHECK(!s.render_time_transform);
  DH_CHECK(s.render_batch_size != 1);
  json config = json::object();
  return add_node("identity", "Identity", config);
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
  DH_CHECK(settings_);
  const auto &s = *settings_;
  json config;
  config["lower_percentile"] = s.view_lower_percentile_;
  config["upper_percentile"] = s.view_upper_percentile_;
  config["radius"] = s.view_reticule_radius_;
  return add_node("percentile_clip", "PercentileClip", config);
}

holoflow::model::DescriptorVertex Worker::add_convert_output_node() {
  json config;
  config["conversion"] = "F32_U8_SCALED";
  return add_node("convert_output", "Convert", config);
}

holoflow::model::DescriptorVertex Worker::add_processed_output_queue_node() {
  accumulators::BatchedSPSCParams config;
  config.nb_slots = 32;
  config.dequeue_batch_size = 1;
  return add_node("processed_output_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Worker::add_processed_display_sink_node() {
  json config = json::object();
  return add_node("processed_display_sink", "QtDisplay", config);
}

} // namespace holovibes::pipeline
