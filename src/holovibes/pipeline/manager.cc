#include "holovibes/pipeline/manager.hh"

#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>

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
#include "holovibes/ui/tensor_display_widget.hh"

using json = nlohmann::json;

namespace holovibes::pipeline {

Manager::Manager(dh::TensorDisplayWidget *processed_display_widget,
                 QObject *parent)
    : QObject(parent), processed_display_widget_(processed_display_widget) {
  dh::holovibes_logger()->debug(
      "[Manager::Manager] Pipeline manager initialized");

  dh::holovibes_logger()->warn(
      "[Manager::Manager] Not all functionalities are implemented yet!");

  // clang-format off
  compiler_.add_factory("Holofile", std::make_unique<dh::HolofileSourceFactory>());
  compiler_.add_factory("QtDisplay", std::make_unique<dh::QtDisplaySinkFactory>(*processed_display_widget_));
  compiler_.add_factory("BatchedSPSC", std::make_unique<dh::BatchedSPSCAccumulatorFactory>());
  compiler_.add_factory("SlidingAverage", std::make_unique<dh::SlidingAverageAccumulatorFactory>());
  compiler_.add_factory("Convert", std::make_unique<dh::ConvertTaskFactory>());
  compiler_.add_factory("Identity", std::make_unique<dh::IdentityTaskFactory>());
  compiler_.add_factory("FresnelDiffraction", std::make_unique<dh::FresnelDiffractionTaskFactory>());
  compiler_.add_factory("AngularSpectrum", std::make_unique<dh::AngularSpectrumTaskFactory>());
  compiler_.add_factory("PCA", std::make_unique<dh::PCATaskFactory>());
  compiler_.add_factory("STFT", std::make_unique<dh::STFTTaskFactory>());
  compiler_.add_factory("Average", std::make_unique<dh::AverageTaskFactory>());
  compiler_.add_factory("PercentileClip", std::make_unique<dh::PercentileClipTaskFactory>());
  compiler_.add_factory("FFTShift", std::make_unique<dh::FFTShiftTaskFactory>());
  // clang-format on
}

void Manager::update_image(const QString &image_type) {
  dh::holovibes_logger()->debug("[Manager::update_image] New image type: {}",
                                image_type.toStdString());

  image_ = image_type.toStdString();
}

void Manager::update_batch_size(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_batch_size] New batch size: {}", value);

  batch_size_ = value;
}

void Manager::update_time_stride(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_time_stride] New time stride: {}", value);

  time_stride_ = value;
}

void Manager::update_filter_2d(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_filter_2d] Filter 2D is: {}",
                                enabled ? "enabled" : "disabled");

  filter_2d_ = enabled;
}

void Manager::update_space_transform(const QString &transform) {
  dh::holovibes_logger()->debug(
      "[Manager::update_space_transform] New space transform: {}",
      transform.toStdString());

  space_transform_ = transform.toStdString();
}

void Manager::update_time_transform(const QString &transform) {
  dh::holovibes_logger()->debug(
      "[Manager::update_time_transform] New time transform: {}",
      transform.toStdString());

  time_transform_ = transform.toStdString();
}

void Manager::update_time_window(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_time_window] New time window: {}", value);

  time_window_ = value;
}

void Manager::update_lambda(int value) {
  dh::holovibes_logger()->debug("[Manager::update_lambda] New lambda: {}",
                                value);

  lambda_ = value;
}

void Manager::update_boundary(int value) {
  dh::holovibes_logger()->debug("[Manager::update_boundary] New boundary: {}",
                                value);

  boundary_ = value;
}

void Manager::update_focus(int value) {
  dh::holovibes_logger()->debug("[Manager::update_focus] New focus: {}", value);

  focus_ = value;
}

void Manager::update_convolution(const QString &convolution_type, bool divide) {
  dh::holovibes_logger()->debug(
      "[Manager::update_convolution] New convolution: {}; Divide: {}",
      convolution_type.toStdString(), divide ? "true" : "false");

  convolution_type_ = convolution_type.toStdString();
  convolution_divide_ = divide;
}

void Manager::update_view_image_type(const QString &image_type) {
  dh::holovibes_logger()->debug(
      "[Manager::update_view_image_type] New view image type: {}",
      image_type.toStdString());

  view_image_type_ = image_type.toStdString();
}

void Manager::update_cuts_3d(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_cuts_3d] Cuts 3D: {}",
                                enabled ? "enabled" : "disabled");

  cuts_3d_ = enabled;
}

void Manager::update_fft_shift(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_fft_shift] FFT Shift: {}",
                                enabled ? "enabled" : "disabled");

  fft_shift_ = enabled;
}

void Manager::update_lens_view(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_lens_view] Lens View: {}",
                                enabled ? "enabled" : "disabled");

  lens_view_ = enabled;
}

void Manager::update_raw_view(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_raw_view] Raw View: {}",
                                enabled ? "enabled" : "disabled");

  raw_view_ = enabled;
}

void Manager::update_z_value(int value) {
  dh::holovibes_logger()->debug("[Manager::update_z_value] New Z value: {}",
                                value);

  z_value_ = value;
}

void Manager::update_width_value(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_width_value] New Width value: {}", value);

  width_value_ = value;
}

void Manager::update_view_kind(const QString &view_kind) {
  dh::holovibes_logger()->debug("[Manager::update_view_kind] New view kind: {}",
                                view_kind.toStdString());

  view_kind_ = view_kind.toStdString();
}

void Manager::update_accumulation(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_accumulation] New accumulation: {}", value);

  accumulation_ = value;
}

void Manager::update_auto(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_auto] Auto: {}",
                                enabled ? "enabled" : "disabled");

  auto_view_ = enabled;
}

void Manager::update_invert(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_invert] Invert: {}",
                                enabled ? "enabled" : "disabled");

  invert_ = enabled;
}

void Manager::update_range_start(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_range_start] New range start: {}", value);

  range_start_ = value;
}

void Manager::update_range_end(int value) {
  dh::holovibes_logger()->debug("[Manager::update_range_end] New range end: {}",
                                value);

  range_end_ = value;
}

void Manager::update_renormalize(bool enabled) {
  dh::holovibes_logger()->debug("[Manager::update_renormalize] Renormalize: {}",
                                enabled ? "enabled" : "disabled");

  renormalize_ = enabled;
}

void Manager::update_import_file(const QString &file_path) {
  dh::holovibes_logger()->debug(
      "[Manager::update_import_file] File selected: {}",
      file_path.toStdString());

  import_file_ = file_path.toStdString();
  if (import_file_ == "") {
    import_file_ = std::nullopt;
  }
}

void Manager::update_import_fps(int value) {
  dh::holovibes_logger()->debug("[Manager::update_import_fps] FPS changed: {}",
                                value);

  import_fps_ = value;
}

void Manager::update_import_start_index(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_import_start_index] Start Index changed: {}", value);

  import_start_index_ = value;
}

void Manager::update_import_end_index(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_import_end_index] End Index changed: {}", value);

  import_end_index_ = value;
}

void Manager::update_import_load_method(const QString &method) {
  dh::holovibes_logger()->debug(
      "[Manager::update_import_load_method] Load method changed: {}",
      method.toStdString());

  import_load_method_ = method.toStdString();
}

void Manager::start_import() {
  dh::holovibes_logger()->debug("[Manager::start_import] Import started");

  build_desc_graph();

  model_ = std::nullopt;
  model_ = std::move(compiler_.compile(desc_graph_));
  runner_ = std::make_unique<holoflow::model::Runner>(*model_);
  runner_->start();
}

void Manager::stop_import() {
  dh::holovibes_logger()->debug("[Manager::stop_import] Import stopped");
  dh::holovibes_logger()->warn("[Manager::stop_import] Not implemented");

  if (!runner_) {
    dh::holovibes_logger()->error(
        "[Manager::stop_import] stop called but model is not running");
  }

  DH_CHECK(model_);
  runner_->stop();

  dh::holovibes_logger()->info("[Manager::stop_import] model stopped");
}

void Manager::update_export_image_type(const QString &image_type) {
  dh::holovibes_logger()->debug(
      "[Manager::update_export_image_type] New export image type: {}",
      image_type.toStdString());

  export_image_type_ = image_type.toStdString();
}

void Manager::update_export_file(const QString &file_path) {
  dh::holovibes_logger()->debug(
      "[Manager::update_export_file] New export file: {}",
      file_path.toStdString());

  export_file_ = file_path.toStdString();
}

void Manager::update_export_tag(const QString &tag) {
  dh::holovibes_logger()->debug(
      "[Manager::update_export_tag] New export tag: {}", tag.toStdString());

  export_tag_ = tag.toStdString();
}

void Manager::update_export_frames_check(bool enabled) {
  dh::holovibes_logger()->debug(
      "[Manager::update_export_frames_check] Export frames check: {}",
      enabled ? "enabled" : "disabled");

  export_frames_check_ = enabled;
}

void Manager::update_export_frames_value(int value) {
  dh::holovibes_logger()->debug(
      "[Manager::update_export_frames_value] New export frames value: {}",
      value);

  export_frames_value_ = value;
}

void Manager::start_export_record() {
  dh::holovibes_logger()->debug(
      "[Manager::start_export_record] Export record started");
  dh::holovibes_logger()->warn(
      "[Manager::start_export_record] Not implemented");
}

void Manager::stop_export() {
  dh::holovibes_logger()->debug("[Manager::stop_export] Export stopped");
  dh::holovibes_logger()->warn("[Manager::stop_export] Not implemented");
}

void Manager::stop_export_fan() {
  dh::holovibes_logger()->debug(
      "[Manager::stop_export_fan] Export stop fan pressed");
  dh::holovibes_logger()->warn("[Manager::stop_export_fan] Not implemented");
}

void Manager::build_desc_graph() {
  desc_graph_ = holoflow::model::DescriptorGraph();

  if (!import_file_) {
    return;
  }

  DH_CHECK(space_transform_);
  DH_CHECK(time_transform_);
  DH_CHECK(fft_shift_);

  auto source_v = add_source_node();
  auto input_queue_v = add_input_queue_node();
  boost::add_edge(source_v, input_queue_v, desc_graph_);
  auto convert_input_v = add_convert_input_node();
  boost::add_edge(input_queue_v, convert_input_v, desc_graph_);

  auto parent_v = convert_input_v;

  if (space_transform_ != "None") {
    auto space_v = add_space_transform_node();
    boost::add_edge(parent_v, space_v, desc_graph_);
    parent_v = space_v;
  }

  if (time_transform_ != "None") {
    auto time_acc_v = add_time_accumulator_node();
    boost::add_edge(parent_v, time_acc_v, desc_graph_);
    auto time_v = add_time_transform_node();
    boost::add_edge(time_acc_v, time_v, desc_graph_);
    parent_v = time_v;
  }

  if (space_transform_ != "None" || time_transform_ != "None") {
    auto convert_postprocess_v = add_convert_postprocess_node();
    boost::add_edge(parent_v, convert_postprocess_v, desc_graph_);
    parent_v = convert_postprocess_v;
  }

  if (time_transform_ != "None") {
    auto time_avg_v = add_p_frame_avg_node();
    boost::add_edge(parent_v, time_avg_v, desc_graph_);
    parent_v = time_avg_v;
  }

  if (*fft_shift_) {
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

  std::ofstream dot_file("pipeline_graph.dot");
  DH_CHECK(dot_file);

  boost::write_graphviz(
      dot_file, desc_graph_,
      holoflow::model::DescriptorNodePropertyWriter(desc_graph_),
      holoflow::model::DescriptorEdgePropertyWriter());

  std::cout << "DOT file 'pipeline_graph.dot' created successfully."
            << std::endl;
}

holoflow::model::DescriptorVertex Manager::add_source_node() {
  DH_CHECK(import_file_);
  DH_CHECK(import_start_index_);
  DH_CHECK(import_end_index_);
  DH_CHECK(batch_size_);
  DH_CHECK(import_load_method_);
  DH_CHECK(import_fps_);
  DH_CHECK(time_stride_);

  json config;
  config["path"] = *import_file_;
  config["start_frame"] = *import_start_index_;
  config["end_frame"] = *import_end_index_;
  config["batch_size"] = *batch_size_;

  if (*import_load_method_ == "Read Live") {
    config["load_kind"] = "READ_LIVE";
  } else if (*import_load_method_ == "Load in CPU RAM") {
    config["load_kind"] = "LOAD_IN_CPU";
  } else if (*import_load_method_ == "Load in GPU RAM") {
    config["load_kind"] = "LOAD_IN_GPU";
  } else {
    DH_BUG("Invalid import load method: \"{}\"", *import_load_method_);
  }

  return add_node("holofile_source", "Holofile", config);
}

holoflow::model::DescriptorVertex Manager::add_input_queue_node() {
  DH_CHECK(batch_size_);

  constexpr size_t TARGET_SIZE = 4096;
  auto nb_slots = TARGET_SIZE / *batch_size_ * *batch_size_;
  json config;
  config["nb_slots"] = nb_slots;
  config["dequeue_batch_size"] = *batch_size_;

  return add_node("input_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Manager::add_convert_input_node() {
  DH_CHECK(space_transform_);
  DH_CHECK(time_transform_);

  auto skip_to_postprocess =
      space_transform_ == "None" && time_transform_ == "None";

  json config;
  config["conversion"] = !skip_to_postprocess ? "U8_CF32_REAL" : "U8_F32";
  return add_node("convert_input", "Convert", config);
}

holoflow::model::DescriptorVertex Manager::add_space_transform_node() {
  DH_CHECK(space_transform_);
  DH_CHECK(space_transform_ != "None");
  DH_CHECK(lambda_);
  DH_CHECK(focus_);

  constexpr float PIXEL_SIZE = 20e-6f;
  json config;
  config["lambda"] = (float)*lambda_ * 1e-9f;
  config["z"] = (float)*focus_ * 1e-3f;
  config["pixel_size"] = PIXEL_SIZE;

  if (space_transform_ == "Fresnel Diffraction") {
    constexpr bool SKIP_PHASE_SHIFT = true;
    config["skip_phase_shift"] = SKIP_PHASE_SHIFT;
    return add_node("fresnel_diffraction", "FresnelDiffraction", config);
  } else if (space_transform_ == "Angular Spectrum") {
    return add_node("angular_spectrum_method", "AngularSpectrum", config);
  } else {
    DH_BUG("Invalid space transform: \"{}\"", *space_transform_);
  }
}

holoflow::model::DescriptorVertex Manager::add_time_accumulator_node() {
  DH_CHECK(time_window_);
  DH_CHECK(batch_size_);
  DH_CHECK(time_transform_);
  DH_CHECK(time_transform_ != "None");

  auto max = std::max(*time_window_, *batch_size_);
  auto target = max * 2 + 1;
  auto gcd_val = std::gcd(*time_window_, *batch_size_);
  auto lcm_val = (*time_window_ / gcd_val) * (*batch_size_);
  auto nb_slots = ((target + lcm_val - 1) / lcm_val) * lcm_val;

  json config;
  config["nb_slots"] = nb_slots;
  config["dequeue_batch_size"] = *time_window_;
  return add_node("time_accumulator", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Manager::add_time_transform_node() {
  DH_CHECK(time_transform_);
  DH_CHECK(time_transform_ != "None");
  DH_CHECK(time_window_);

  json config;
  if (time_transform_ == "Principal Component Analysis") {
    DH_CHECK(z_value_);
    DH_CHECK(width_value_);
    auto begin = *z_value_;
    auto end = begin + *width_value_;
    config["begin"] = begin;
    config["end"] = end;
    return add_node("principal_component_analysis", "PCA", config);
  } else if (time_transform_ == "Short Time Fourrier Transform") {
    return add_node("short_time_fourrier_transform", "STFT", config);
  } else {
    DH_BUG("Invalid time transform: \"{}\"", *time_transform_);
  }
}

holoflow::model::DescriptorVertex Manager::add_convert_postprocess_node() {
  DH_CHECK(space_transform_);
  DH_CHECK(time_transform_);
  DH_CHECK(space_transform_ != "None" || time_transform_ != "None");

  json config;
  config["conversion"] = "CF32_F32_MODU";
  return add_node("convert_postprocess", "Convert", config);
}

holoflow::model::DescriptorVertex Manager::add_p_frame_avg_node() {
  DH_CHECK(z_value_);
  DH_CHECK(width_value_);
  DH_CHECK(time_transform_);
  DH_CHECK(time_transform_ != "None");

  json config;
  auto begin = *z_value_;
  auto end = begin + *width_value_;
  config["begin"] = begin;
  config["end"] = end;
  return add_node("p_frame_average", "Average", config);
}

holoflow::model::DescriptorVertex Manager::add_fft_shift_node() {
  DH_CHECK(fft_shift_);
  DH_CHECK(*fft_shift_ == true);

  return add_node("fft_shift", "FFTShift", R"({})"_json);
}

holoflow::model::DescriptorVertex Manager::add_image_avg_accumulator_node() {
  DH_CHECK(accumulation_);

  json config;
  config["window_size"] = *accumulation_;
  config["nb_slots"] = *accumulation_ + 10;
  return add_node("image_avg_accumulator", "SlidingAverage", config);
}

holoflow::model::DescriptorVertex Manager::add_percentile_clip_node() {
  json config;
  config["lower_percentile"] = 0.1f;
  config["upper_percentile"] = 99.9f;
  return add_node("percentile_clip", "PercentileClip", config);
}

holoflow::model::DescriptorVertex Manager::add_convert_output_node() {
  json config;
  config["conversion"] = "F32_U8_SCALED";
  return add_node("convert_output", "Convert", config);
}

holoflow::model::DescriptorVertex Manager::add_processed_output_queue_node() {
  json config;
  config["nb_slots"] = 32;
  config["dequeue_batch_size"] = 1;
  return add_node("processed_output_queue", "BatchedSPSC", config);
}

holoflow::model::DescriptorVertex Manager::add_processed_display_sink_node() {
  return add_node("processed_display_sink", "QtDisplay", R"({})"_json);
}

holoflow::model::DescriptorVertex Manager::add_node(const std::string &id,
                                                    const std::string &type,
                                                    const json &config) {
  holoflow::model::DescriptorVertex v = boost::add_vertex(desc_graph_);
  desc_graph_[v].id = id;
  desc_graph_[v].type = type;
  desc_graph_[v].config = config;
  nodes_[id] = v;
  return v;
}

void Manager::add_edge_by_id(const std::string &src, const std::string &dst) {
  boost::add_edge(nodes_[src], nodes_[dst], desc_graph_);
}

std::optional<holoflow::model::DescriptorVertex>
Manager::find_node_by_id(const std::string &id) {
  auto [start, end] = boost::vertices(desc_graph_);
  auto it = std::find_if(
      start, end, [this, &id](auto v) { return desc_graph_[v].id == id; });

  return it != end ? std::make_optional(*it) : std::nullopt;
}

} // namespace holovibes::pipeline