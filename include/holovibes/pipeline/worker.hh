#pragma once

#include <QObject>
#include <concepts>
#include <optional>

#include "holoflow/v3/model/compiler.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/model.hh"
#include "holoflow/v3/model/runner.hh"
#include "holovibes/pipeline/settings.hh"
#include "holovibes/ui/tensor_display_widget.hh"

namespace holovibes::pipeline {

template <typename T>
concept ToJson = requires(const T &t) {
  { nlohmann::json(t) } -> std::same_as<nlohmann::json>;
};

class Worker : public QObject {
  Q_OBJECT

public:
  explicit Worker(dh::TensorDisplayWidget *processed_display_widget,
                  dh::TensorDisplayWidget *raw_record_display_widget_,
                  QObject *parent = nullptr);

  void set_settings(const Settings &settings);

public slots:
  void start();
  void stop();
  void update();
  void start_export();
  void stop_export();

signals:
  void start_success();
  void start_failure();
  void stop_success();
  void stop_failure();
  void update_success();
  void update_failure();
  void start_export_success();
  void start_export_failure();
  void stop_export_success();
  void stop_export_failure();

private:
  void build_desc_graph();

  holoflow::model::DescriptorVertex add_node(const std::string &id,
                                             const std::string &type,
                                             const nlohmann::json &config);
  template <ToJson Config>
  holoflow::model::DescriptorVertex add_node(const std::string &id,
                                             const std::string &type,
                                             const Config &config) {
    return add_node(id, type, nlohmann::json(config));
  }

  holoflow::model::DescriptorVertex add_source_node();
  holoflow::model::DescriptorVertex add_raw_record_gate_node();
  holoflow::model::DescriptorVertex add_raw_identity_node();
  holoflow::model::DescriptorVertex add_raw_record_accumulator_node();
  holoflow::model::DescriptorVertex add_raw_record_display_sink_node();
  holoflow::model::DescriptorVertex add_cpu_input_queue_node();
  holoflow::model::DescriptorVertex add_cpy_cpu_to_gpu_node();
  holoflow::model::DescriptorVertex add_input_queue_node();
  holoflow::model::DescriptorVertex add_convert_input_node();
  holoflow::model::DescriptorVertex add_space_transform_node();
  holoflow::model::DescriptorVertex add_time_accumulator_node();
  holoflow::model::DescriptorVertex add_time_transform_node();
  holoflow::model::DescriptorVertex add_convert_postprocess_node();
  holoflow::model::DescriptorVertex add_p_frame_avg_node();
  holoflow::model::DescriptorVertex add_fft_shift_node();
  holoflow::model::DescriptorVertex add_split_axis_0_node();
  holoflow::model::DescriptorVertex add_identity_node();
  holoflow::model::DescriptorVertex add_image_avg_accumulator_node();
  holoflow::model::DescriptorVertex add_percentile_clip_node();
  holoflow::model::DescriptorVertex add_convert_output_node();
  holoflow::model::DescriptorVertex add_processed_output_queue_node();
  holoflow::model::DescriptorVertex add_processed_display_sink_node();

  // External widgets
  dh::TensorDisplayWidget *processed_display_widget_;
  dh::TensorDisplayWidget *raw_record_display_widget_;

  // Settings
  std::optional<Settings> settings_;

  // Pipeline components
  holoflow::model::ModelCompiler compiler_;
  holoflow::model::DescriptorGraph desc_graph_;
  holoflow::model::ModelCompiler::EventListenerMap event_listeners_;
  std::unordered_map<std::string, holoflow::model::DescriptorVertex> nodes_;
  std::optional<holoflow::model::Model> model_;
  std::unique_ptr<holoflow::model::Runner> runner_;

  // Optimization flags
  bool opti_early_stride_;
};

} // namespace holovibes::pipeline