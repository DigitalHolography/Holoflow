#pragma once

#include <QObject>
#include <QString>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "holoflow/v3/model/descriptor.hh"

using json = nlohmann::json;

namespace holovibes::pipeline {

class Manager : public QObject {
  Q_OBJECT

public:
  explicit Manager(QObject *parent = nullptr);

public slots:
  // Image rendering slots
  void update_image(const QString &image_type);
  void update_batch_size(int value);
  void update_time_stride(int value);
  void update_filter_2d(bool enabled);
  void update_space_transform(const QString &transform);
  void update_time_transform(const QString &transform);
  void update_time_window(int value);
  void update_lambda(int value);
  void update_boundary(int value);
  void update_focus(int value);
  void update_convolution(const QString &convolution_type, bool divide);

  // Image view slots
  void update_view_image_type(const QString &image_type);
  void update_cuts_3d(bool enabled);
  void update_fft_shift(bool enabled);
  void update_lens_view(bool enabled);
  void update_raw_view(bool enabled);
  void update_z_value(int value);
  void update_width_value(int value);
  void update_view_kind(const QString &view_kind);
  void update_accumulation(int value);
  void update_auto(bool enabled);
  void update_invert(bool enabled);
  void update_range_start(int value);
  void update_range_end(int value);
  void update_renormalize(bool enabled);

  // Import slots
  void update_import_file(const QString &file_path);
  void update_import_fps(int value);
  void update_import_start_index(int value);
  void update_import_end_index(int value);
  void update_import_load_method(const QString &method);
  void start_import();
  void stop_import();

  // Export slots
  void update_export_image_type(const QString &image_type);
  void update_export_file(const QString &file_path);
  void update_export_tag(const QString &tag);
  void update_export_frames_check(bool enabled);
  void update_export_frames_value(int value);
  void start_export_record();
  void stop_export();
  void stop_export_fan();

private:
  void build_desc_graph();

  holoflow::model::DescriptorVertex add_source_node();
  holoflow::model::DescriptorVertex add_input_queue_node();
  holoflow::model::DescriptorVertex add_convert_input_node();
  holoflow::model::DescriptorVertex add_space_transform_node();
  holoflow::model::DescriptorVertex add_time_accumulator_node();
  holoflow::model::DescriptorVertex add_time_transform_node();

  holoflow::model::DescriptorVertex
  add_node(const std::string &id, const std::string &type, const json &config);

  void add_edge_by_id(const std::string &src, const std::string &dst);

  std::optional<holoflow::model::DescriptorVertex>
  find_node_by_id(const std::string &id);

  holoflow::model::DescriptorGraph desc_graph_;
  std::map<std::string, holoflow::model::DescriptorVertex> nodes_;

  // Copies of Image Rendering Settings
  std::optional<std::string> image_;
  std::optional<int> batch_size_;
  std::optional<int> time_stride_;
  std::optional<bool> filter_2d_;
  std::optional<std::string> space_transform_;
  std::optional<std::string> time_transform_;
  std::optional<int> time_window_;
  std::optional<int> lambda_;
  std::optional<int> boundary_;
  std::optional<int> focus_;
  std::optional<std::string> convolution_type_;
  std::optional<bool> convolution_divide_;

  // Copies of Image View Settings
  std::optional<std::string> view_image_type_;
  std::optional<bool> cuts_3d_;
  std::optional<bool> fft_shift_;
  std::optional<bool> lens_view_;
  std::optional<bool> raw_view_;
  std::optional<int> z_value_;
  std::optional<int> width_value_;
  std::optional<std::string> view_kind_;
  std::optional<int> accumulation_;
  std::optional<bool> auto_view_;
  std::optional<bool> invert_;
  std::optional<int> range_start_;
  std::optional<int> range_end_;
  std::optional<bool> renormalize_;

  // Copies of Import Settings
  std::optional<std::string> import_file_;
  std::optional<int> import_fps_;
  std::optional<int> import_start_index_;
  std::optional<int> import_end_index_;
  std::optional<std::string> import_load_method_;

  // Copies of Export Settings
  std::optional<std::string> export_image_type_;
  std::optional<std::string> export_file_;
  std::optional<std::string> export_tag_;
  std::optional<bool> export_frames_check_;
  std::optional<int> export_frames_value_;
};

} // namespace holovibes::pipeline