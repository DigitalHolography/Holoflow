#include "holovibes/ui/main_window.hh"

#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QSlider>
#include <QSpacerItem>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include "holovibes/pipeline/manager.hh"
#include "holovibes/ui/export_widget.hh"
#include "holovibes/ui/image_rendering_widget.hh"
#include "holovibes/ui/import_widget.hh"
#include "holovibes/ui/tensor_display_widget.hh"
#include "holovibes/ui/view_widget.hh"

namespace holovibes::ui {

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  processed_display_widget_ = new dh::TensorDisplayWidget(800, 800);
  processed_display_widget_->show();

  // Create menus
  menuBar()->addMenu("&File");
  menuBar()->addMenu("&View");
  menuBar()->addMenu("&Camera");
  menuBar()->addMenu("&Theme");

  // Set up central widget
  auto *central_widget = new QWidget(this);
  setCentralWidget(central_widget);
  auto *main_layout = new QHBoxLayout(central_widget);

  // Create left panel layout
  auto *left_panel_layout = new QHBoxLayout();

  auto *image_rendering_widget = new ImageRenderingWidget(this);
  left_panel_layout->addWidget(image_rendering_widget);

  auto *view_widget = new ViewWidget(this);
  left_panel_layout->addWidget(view_widget);

  auto *import_export_layout = new QVBoxLayout();
  auto *import_widget = new ImportWidget(this);
  import_export_layout->addWidget(import_widget);

  auto *export_widget = new ExportWidget(this);
  import_export_layout->addWidget(export_widget);
  left_panel_layout->addLayout(import_export_layout);

  main_layout->addLayout(left_panel_layout);

  setWindowTitle("HoloVibes");
  resize(1000, 600);

  auto *pipeline_mgr = new pipeline::Manager(processed_display_widget_, this);

  // Link Image Rendering widget signals to Pipeline Manager.
  connect(image_rendering_widget, &ImageRenderingWidget::image_changed,
          pipeline_mgr, &pipeline::Manager::update_image);
  connect(image_rendering_widget, &ImageRenderingWidget::batch_size_changed,
          pipeline_mgr, &pipeline::Manager::update_batch_size);
  connect(image_rendering_widget, &ImageRenderingWidget::time_stride_changed,
          pipeline_mgr, &pipeline::Manager::update_time_stride);
  connect(image_rendering_widget, &ImageRenderingWidget::filter_2d_toggled,
          pipeline_mgr, &pipeline::Manager::update_filter_2d);
  connect(image_rendering_widget,
          &ImageRenderingWidget::space_transform_changed, pipeline_mgr,
          &pipeline::Manager::update_space_transform);
  connect(image_rendering_widget, &ImageRenderingWidget::time_transform_changed,
          pipeline_mgr, &pipeline::Manager::update_time_transform);
  connect(image_rendering_widget, &ImageRenderingWidget::time_window_changed,
          pipeline_mgr, &pipeline::Manager::update_time_window);
  connect(image_rendering_widget, &ImageRenderingWidget::lambda_changed,
          pipeline_mgr, &pipeline::Manager::update_lambda);
  connect(image_rendering_widget, &ImageRenderingWidget::boundary_changed,
          pipeline_mgr, &pipeline::Manager::update_boundary);
  connect(image_rendering_widget, &ImageRenderingWidget::focus_changed,
          pipeline_mgr, &pipeline::Manager::update_focus);
  connect(image_rendering_widget, &ImageRenderingWidget::focus_slider_changed,
          pipeline_mgr, &pipeline::Manager::update_focus);
  connect(image_rendering_widget, &ImageRenderingWidget::convolution_changed,
          pipeline_mgr, &pipeline::Manager::update_convolution);

  // Link View widget signals to Pipeline Manager.
  connect(view_widget, &ViewWidget::image_type_changed, pipeline_mgr,
          &pipeline::Manager::update_view_image_type);
  connect(view_widget, &ViewWidget::cuts_3d_toggled, pipeline_mgr,
          &pipeline::Manager::update_cuts_3d);
  connect(view_widget, &ViewWidget::fft_shift_toggled, pipeline_mgr,
          &pipeline::Manager::update_fft_shift);
  connect(view_widget, &ViewWidget::lens_view_toggled, pipeline_mgr,
          &pipeline::Manager::update_lens_view);
  connect(view_widget, &ViewWidget::raw_view_toggled, pipeline_mgr,
          &pipeline::Manager::update_raw_view);
  connect(view_widget, &ViewWidget::z_changed, pipeline_mgr,
          &pipeline::Manager::update_z_value);
  connect(view_widget, &ViewWidget::width_changed, pipeline_mgr,
          &pipeline::Manager::update_width_value);
  connect(view_widget, &ViewWidget::view_kind_changed, pipeline_mgr,
          &pipeline::Manager::update_view_kind);
  connect(view_widget, &ViewWidget::accumulation_changed, pipeline_mgr,
          &pipeline::Manager::update_accumulation);
  connect(view_widget, &ViewWidget::auto_changed, pipeline_mgr,
          &pipeline::Manager::update_auto);
  connect(view_widget, &ViewWidget::invert_changed, pipeline_mgr,
          &pipeline::Manager::update_invert);
  connect(view_widget, &ViewWidget::range_start_changed, pipeline_mgr,
          &pipeline::Manager::update_range_start);
  connect(view_widget, &ViewWidget::range_end_changed, pipeline_mgr,
          &pipeline::Manager::update_range_end);
  connect(view_widget, &ViewWidget::renormalize_changed, pipeline_mgr,
          &pipeline::Manager::update_renormalize);

  // Link Import widget signals to Pipeline Manager.
  connect(import_widget, &ImportWidget::file_selected, pipeline_mgr,
          &pipeline::Manager::update_import_file);
  connect(import_widget, &ImportWidget::fps_changed, pipeline_mgr,
          &pipeline::Manager::update_import_fps);
  connect(import_widget, &ImportWidget::start_index_changed, pipeline_mgr,
          &pipeline::Manager::update_import_start_index);
  connect(import_widget, &ImportWidget::end_index_changed, pipeline_mgr,
          &pipeline::Manager::update_import_end_index);
  connect(import_widget, &ImportWidget::load_method_changed, pipeline_mgr,
          &pipeline::Manager::update_import_load_method);
  connect(import_widget, &ImportWidget::start_import, pipeline_mgr,
          &pipeline::Manager::start_import);
  connect(import_widget, &ImportWidget::stop_import, pipeline_mgr,
          &pipeline::Manager::stop_import);

  // Link Export widget signals.
  connect(export_widget, &ExportWidget::export_image_type_changed, pipeline_mgr,
          &pipeline::Manager::update_export_image_type);
  connect(export_widget, &ExportWidget::export_file_selected, pipeline_mgr,
          &pipeline::Manager::update_export_file);
  connect(export_widget, &ExportWidget::export_tag_changed, pipeline_mgr,
          &pipeline::Manager::update_export_tag);
  connect(export_widget, &ExportWidget::export_frames_check_changed,
          pipeline_mgr, &pipeline::Manager::update_export_frames_check);
  connect(export_widget, &ExportWidget::export_frames_value_changed,
          pipeline_mgr, &pipeline::Manager::update_export_frames_value);
  connect(export_widget, &ExportWidget::export_record_pressed, pipeline_mgr,
          &pipeline::Manager::start_export_record);
  connect(export_widget, &ExportWidget::export_stop_pressed, pipeline_mgr,
          &pipeline::Manager::stop_export);
  connect(export_widget, &ExportWidget::export_stop_fan_pressed, pipeline_mgr,
          &pipeline::Manager::stop_export_fan);

  // Push the initial values using a singleShot timer so that everything is set
  // up.
  QTimer::singleShot(0, this, [=]() {
    // For Image Rendering widget.
    pipeline_mgr->update_image(image_rendering_widget->current_image());
    pipeline_mgr->update_batch_size(
        image_rendering_widget->current_batch_size());
    pipeline_mgr->update_time_stride(
        image_rendering_widget->current_time_stride());
    pipeline_mgr->update_filter_2d(image_rendering_widget->current_filter_2d());
    pipeline_mgr->update_space_transform(
        image_rendering_widget->current_space_transform());
    pipeline_mgr->update_time_transform(
        image_rendering_widget->current_time_transform());
    pipeline_mgr->update_time_window(
        image_rendering_widget->current_time_window());
    pipeline_mgr->update_lambda(image_rendering_widget->current_lambda());
    pipeline_mgr->update_boundary(image_rendering_widget->current_boundary());
    pipeline_mgr->update_focus(image_rendering_widget->current_focus());
    pipeline_mgr->update_convolution(
        image_rendering_widget->current_convolution(),
        image_rendering_widget->current_convolution_divide());

    // For View widget.
    pipeline_mgr->update_view_image_type(view_widget->current_image_type());
    pipeline_mgr->update_cuts_3d(view_widget->current_cuts_3d());
    pipeline_mgr->update_fft_shift(view_widget->current_fft_shift());
    pipeline_mgr->update_lens_view(view_widget->current_lens_view());
    pipeline_mgr->update_raw_view(view_widget->current_raw_view());
    pipeline_mgr->update_z_value(view_widget->current_z_value());
    pipeline_mgr->update_width_value(view_widget->current_width_value());
    pipeline_mgr->update_view_kind(view_widget->current_view_kind());
    pipeline_mgr->update_accumulation(view_widget->current_accumulation());
    pipeline_mgr->update_auto(view_widget->current_auto());
    pipeline_mgr->update_invert(view_widget->current_invert());
    pipeline_mgr->update_range_start(view_widget->current_range_start());
    pipeline_mgr->update_range_end(view_widget->current_range_end());
    pipeline_mgr->update_renormalize(view_widget->current_renormalize());

    // For Import widget.
    pipeline_mgr->update_import_file(import_widget->current_file());
    pipeline_mgr->update_import_fps(import_widget->current_fps());
    pipeline_mgr->update_import_start_index(
        import_widget->current_start_index());
    pipeline_mgr->update_import_end_index(import_widget->current_end_index());
    pipeline_mgr->update_import_load_method(
        import_widget->current_load_method());

    // For Export widget.
    pipeline_mgr->update_export_image_type(export_widget->current_image_type());
    pipeline_mgr->update_export_file(export_widget->current_file());
    pipeline_mgr->update_export_tag(export_widget->current_tag());
    pipeline_mgr->update_export_frames_check(
        export_widget->current_frames_check());
    pipeline_mgr->update_export_frames_value(
        export_widget->current_frames_value());
  });
}

} // namespace holovibes::ui