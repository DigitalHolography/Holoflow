// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ui/main_window.hh"

#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QDrag>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfoList>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QMouseEvent>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSpacerItem>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedLayout>
#include <QStandardPaths>
#include <QStringList>
#include <QStyle>
#include <QThread>
#include <QVBoxLayout>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>

#include "bug.hh"
#include "holofile/holofile.hh"
#include "logger.hh"
#include "settings_loader.hh"
#include "ui/widgets/tensor_display_widget.hh"

namespace {

constexpr int  kLargeSpinMax             = 1024 * 1024;
constexpr int  kTopBarHeight             = 30;
constexpr int  kEmptySecondaryDropHeight = 72;
constexpr auto kDisplayPanelMimeType     = "application/x-holovibes-display-panel";
constexpr auto kDisplayPanelIdProperty   = "displayPanelId";
constexpr auto kDisplayPanelMainProperty = "displayPanelInMain";

QSpinBox *create_spin_box(QWidget *parent, int minimum, int maximum, int value) {
  auto *spin_box = new QSpinBox(parent);
  spin_box->setRange(minimum, maximum);
  spin_box->setValue(value);
  return spin_box;
}

QDoubleSpinBox *create_double_spin_box(QWidget *parent, double minimum, double maximum, double step,
                                       double value, int decimals = 2) {
  auto *spin_box = new QDoubleSpinBox(parent);
  spin_box->setRange(minimum, maximum);
  spin_box->setSingleStep(step);
  spin_box->setDecimals(decimals);
  spin_box->setValue(value);
  return spin_box;
}

QComboBox *create_combo_box(QWidget *parent, const QStringList &items) {
  auto *combo_box = new QComboBox(parent);
  combo_box->addItems(items);
  return combo_box;
}

struct FieldBinding {
  holovibes::pipeline::SettingsField field;
  QWidget                           *widget;
};

QString to_qstring(holovibes::pipeline::ValidationSeverity severity) {
  switch (severity) {
  case holovibes::pipeline::ValidationSeverity::Warning:
    return "Warning";
  case holovibes::pipeline::ValidationSeverity::Error:
    return "Error";
  }

  HOLOVIBES_UNREACHABLE();
}

QString build_field_tooltip(const holovibes::pipeline::FieldHelp                        &help,
                            std::span<const holovibes::pipeline::ValidationIssue *const> issues) {
  QStringList lines;
  lines << QString::fromStdString(help.title);
  lines << "";
  lines << QString::fromStdString(help.description);

  if (!help.constraints.empty()) {
    lines << "";
    lines << "Constraints:";
    for (const auto *constraint : help.constraints) {
      lines << QString("- %1").arg(constraint);
    }
  }

  if (!issues.empty()) {
    lines << "";
    lines << "Current issues:";
    for (const auto *issue : issues) {
      lines << QString("- %1: %2")
                   .arg(to_qstring(issue->severity), QString::fromStdString(issue->message));
    }
  }

  return lines.join('\n');
}

void restore_combo_text(QSettings &settings, const char *key, QComboBox *combo_box) {
  const QString text = settings.value(key).toString();
  if (text.isEmpty()) {
    return;
  }

  const int index = combo_box->findText(text);
  if (index >= 0) {
    combo_box->setCurrentIndex(index);
  }
}

void clear_layout(QLayout *layout) {
  auto     *grid_layout = qobject_cast<QGridLayout *>(layout);
  const int rows        = grid_layout != nullptr ? grid_layout->rowCount() : 0;
  const int columns     = grid_layout != nullptr ? grid_layout->columnCount() : 0;

  while (auto *item = layout->takeAt(0)) {
    delete item;
  }

  if (grid_layout != nullptr) {
    for (int row = 0; row < rows; ++row) {
      grid_layout->setRowStretch(row, 0);
      grid_layout->setRowMinimumHeight(row, 0);
    }
    for (int column = 0; column < columns; ++column) {
      grid_layout->setColumnStretch(column, 0);
      grid_layout->setColumnMinimumWidth(column, 0);
    }
  }
}

void repolish(QWidget *widget) {
  widget->style()->unpolish(widget);
  widget->style()->polish(widget);
  widget->update();
}

void set_drag_active(QWidget *widget, bool active) {
  widget->setProperty("dragActive", active);
  repolish(widget);
}

void start_display_panel_drag(QWidget *source, const QString &display_id) {
  auto *mime_data = new QMimeData();
  mime_data->setData(kDisplayPanelMimeType, display_id.toUtf8());

  auto *drag = new QDrag(source);
  drag->setMimeData(mime_data);
  drag->exec(Qt::MoveAction);
}

bool accepts_display_panel_drag(QDragEnterEvent *event) {
  if (!event->mimeData()->hasFormat(kDisplayPanelMimeType)) {
    return false;
  }

  event->acceptProposedAction();
  return true;
}

bool accepts_display_panel_drag(QDragMoveEvent *event) {
  if (!event->mimeData()->hasFormat(kDisplayPanelMimeType)) {
    return false;
  }

  event->acceptProposedAction();
  return true;
}

QString dropped_display_panel_id(QDropEvent *event) {
  if (!event->mimeData()->hasFormat(kDisplayPanelMimeType)) {
    return {};
  }

  event->acceptProposedAction();
  return QString::fromUtf8(event->mimeData()->data(kDisplayPanelMimeType));
}

class DraggableDisplayPanel : public QGroupBox {
public:
  using DropHandler = std::function<void(const QString &, bool)>;

  DraggableDisplayPanel(const QString &title, const QString &display_id, DropHandler drop_handler,
                        QWidget *parent)
      : QGroupBox(title, parent), display_id_(display_id), drop_handler_(std::move(drop_handler)) {
    setObjectName("displayPanel");
    setAcceptDrops(true);
    setProperty(kDisplayPanelIdProperty, display_id_);
    setProperty(kDisplayPanelMainProperty, false);
  }

protected:
  void mousePressEvent(QMouseEvent *event) override {
    if (event->button() == Qt::LeftButton) {
      drag_start_pos_ = event->pos();
    }

    QGroupBox::mousePressEvent(event);
  }

  void mouseMoveEvent(QMouseEvent *event) override {
    if ((event->buttons() & Qt::LeftButton) != 0 &&
        (event->pos() - drag_start_pos_).manhattanLength() >= QApplication::startDragDistance()) {
      start_display_panel_drag(this, display_id_);
      return;
    }

    QGroupBox::mouseMoveEvent(event);
  }

  void dragEnterEvent(QDragEnterEvent *event) override {
    if (accepts_display_panel_drag(event)) {
      set_drag_active(this, true);
    }
  }

  void dragMoveEvent(QDragMoveEvent *event) override { accepts_display_panel_drag(event); }

  void dragLeaveEvent(QDragLeaveEvent *event) override {
    set_drag_active(this, false);
    QGroupBox::dragLeaveEvent(event);
  }

  void dropEvent(QDropEvent *event) override {
    const QString dropped_id = dropped_display_panel_id(event);
    set_drag_active(this, false);
    if (dropped_id.isEmpty() || dropped_id == display_id_) {
      return;
    }

    drop_handler_(dropped_id, property(kDisplayPanelMainProperty).toBool());
  }

private:
  QString     display_id_;
  DropHandler drop_handler_;
  QPoint      drag_start_pos_;
};

class SquareDisplayViewport : public QFrame {
public:
  using DropHandler = DraggableDisplayPanel::DropHandler;

  SquareDisplayViewport(holovibes::ui::TensorDisplayWidget *display, const QString &display_id,
                        DropHandler drop_handler, QWidget *parent)
      : QFrame(parent), display_(display), display_id_(display_id),
        drop_handler_(std::move(drop_handler)) {
    setObjectName("displayViewport");
    setAcceptDrops(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setMinimumSize(0, 0);

    display_->setParent(this);
    display_->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    display_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    display_->setMinimumSize(0, 0);
    display_->show();
  }

protected:
  void mousePressEvent(QMouseEvent *event) override {
    if (event->button() == Qt::LeftButton) {
      drag_start_pos_ = event->pos();
    }

    QFrame::mousePressEvent(event);
  }

  void mouseMoveEvent(QMouseEvent *event) override {
    if ((event->buttons() & Qt::LeftButton) != 0 &&
        (event->pos() - drag_start_pos_).manhattanLength() >= QApplication::startDragDistance()) {
      start_display_panel_drag(this, display_id_);
      return;
    }

    QFrame::mouseMoveEvent(event);
  }

  void resizeEvent(QResizeEvent *event) override {
    QFrame::resizeEvent(event);

    const int side = std::max(0, std::min(width(), height()));
    const int x    = (width() - side) / 2;
    const int y    = (height() - side) / 2;
    display_->setGeometry(x, y, side, side);
  }

  void dragEnterEvent(QDragEnterEvent *event) override {
    if (accepts_display_panel_drag(event)) {
      set_drag_active(this, true);
    }
  }

  void dragMoveEvent(QDragMoveEvent *event) override { accepts_display_panel_drag(event); }

  void dragLeaveEvent(QDragLeaveEvent *event) override {
    set_drag_active(this, false);
    QFrame::dragLeaveEvent(event);
  }

  void dropEvent(QDropEvent *event) override {
    const QString dropped_id = dropped_display_panel_id(event);
    set_drag_active(this, false);
    if (dropped_id.isEmpty() || dropped_id == display_id_) {
      return;
    }

    auto *panel = qobject_cast<QWidget *>(parent());
    if (panel == nullptr) {
      return;
    }

    drop_handler_(dropped_id, panel->property(kDisplayPanelMainProperty).toBool());
  }

private:
  holovibes::ui::TensorDisplayWidget *display_;
  QString                             display_id_;
  DropHandler                         drop_handler_;
  QPoint                              drag_start_pos_;
};

class DisplayDropZone : public QFrame {
public:
  using DropHandler = std::function<void(const QString &)>;

  explicit DisplayDropZone(DropHandler drop_handler, QWidget *parent)
      : QFrame(parent), drop_handler_(std::move(drop_handler)) {
    setAcceptDrops(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

protected:
  void dragEnterEvent(QDragEnterEvent *event) override {
    if (accepts_display_panel_drag(event)) {
      set_drag_active(this, true);
    }
  }

  void dragMoveEvent(QDragMoveEvent *event) override { accepts_display_panel_drag(event); }

  void dragLeaveEvent(QDragLeaveEvent *event) override {
    set_drag_active(this, false);
    QFrame::dragLeaveEvent(event);
  }

  void dropEvent(QDropEvent *event) override {
    const QString dropped_id = dropped_display_panel_id(event);
    set_drag_active(this, false);
    if (!dropped_id.isEmpty()) {
      drop_handler_(dropped_id);
    }
  }

private:
  DropHandler drop_handler_;
};

} // namespace

namespace holovibes::ui {

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), session_id_(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")) {
  setup_main_layout();
  initialize_display_widgets();
  restore_persistent_state();
  configure_unsupported_features();
  initialize_pipeline_manager();

  connect_manager_signals();
  connect_import_controls();
  connect_export_controls();
  setup_validation_connections();
  setup_update_connections();

  validate_inputs();
  refresh_command_bar();
  configure_window();
}

void MainWindow::setup_main_layout() {
  auto *central_widget = new QWidget(this);
  setCentralWidget(central_widget);

  auto *main_layout = new QVBoxLayout(central_widget);
  main_layout->setContentsMargins(8, 6, 8, 8);
  main_layout->setSpacing(4);

  auto *session_bar = new QFrame(central_widget);
  session_bar->setObjectName("sessionBar");
  session_bar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  session_bar->setFixedHeight(kTopBarHeight);
  auto *session_layout = new QHBoxLayout(session_bar);
  session_layout->setContentsMargins(8, 2, 8, 2);
  session_layout->setSpacing(6);

  auto *patient_label = new QLabel("Patient", session_bar);
  patient_line_edit_  = new QLineEdit(session_bar);
  patient_line_edit_->setObjectName("patientField");
  patient_line_edit_->setText("PATIENT");
  patient_line_edit_->setPlaceholderText("PATIENT");
  patient_line_edit_->setMinimumWidth(180);
  patient_line_edit_->setMaximumWidth(260);

  auto *eye_label = new QLabel("Eye", session_bar);
  eye_side_combo_ = create_combo_box(session_bar, QStringList{"NA", "OD", "OS", "OU"});
  eye_side_combo_->setObjectName("eyeSideField");
  eye_side_combo_->setToolTip(
      "Eye side used in recording file names. OD: right eye, OS: left eye, OU: both eyes, NA: not "
      "specified.");

  auto *session_label  = new QLabel("Session", session_bar);
  session_value_label_ = new QLabel(session_id_, session_bar);
  session_value_label_->setObjectName("sessionValue");

  auto *next_acquisition_label = new QLabel("Next Acquisition", session_bar);
  acquisition_value_label_     = new QLabel(acquisition_label(next_acquisition_id_), session_bar);
  acquisition_value_label_->setObjectName("sessionValue");

  session_layout->addWidget(patient_label);
  session_layout->addWidget(patient_line_edit_);
  session_layout->addSpacing(8);
  session_layout->addWidget(eye_label);
  session_layout->addWidget(eye_side_combo_);
  session_layout->addSpacing(16);
  session_layout->addWidget(session_label);
  session_layout->addWidget(session_value_label_);
  session_layout->addSpacing(16);
  session_layout->addWidget(next_acquisition_label);
  session_layout->addWidget(acquisition_value_label_);
  session_layout->addStretch(1);
  main_layout->addWidget(session_bar, 0);

  render_widget_ = new ImageRenderingWidget(this);
  view_widget_   = new ViewWidget(this);
  import_widget_ = new ImportWidget(this);
  export_widget_ = new ExportWidget(this);

  auto *command_bar = new QFrame(central_widget);
  command_bar->setObjectName("commandBar");
  command_bar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  command_bar->setFixedHeight(kTopBarHeight);
  auto *command_layout = new QHBoxLayout(command_bar);
  command_layout->setContentsMargins(8, 2, 8, 2);
  command_layout->setSpacing(6);

  start_command_button_       = new QPushButton("Start", command_bar);
  stop_command_button_        = new QPushButton("Stop", command_bar);
  record_command_button_      = new QPushButton("Record", command_bar);
  stop_record_command_button_ = new QPushButton("Stop Rec", command_bar);
  start_command_button_->setObjectName("primaryCommand");
  record_command_button_->setObjectName("recordCommand");

  command_layout->addWidget(start_command_button_);
  command_layout->addWidget(stop_command_button_);
  command_layout->addSpacing(8);
  command_layout->addWidget(record_command_button_);
  command_layout->addWidget(stop_record_command_button_);
  command_layout->addSpacing(16);

  auto add_status_field = [&](const QString &label, QLabel *&value_label) {
    auto *name_label = new QLabel(label, command_bar);
    name_label->setObjectName("commandLabel");
    value_label = new QLabel(command_bar);
    value_label->setObjectName("commandValue");
    command_layout->addWidget(name_label);
    command_layout->addWidget(value_label);
    command_layout->addSpacing(12);
  };

  add_status_field("Status", pipeline_status_label_);
  add_status_field("Source", source_status_label_);
  add_status_field("View", view_status_label_);
  add_status_field("FPS", fps_status_label_);
  add_status_field("Recording", recording_status_label_);
  command_layout->addStretch(1);
  main_layout->addWidget(command_bar, 0);

  connect(start_command_button_, &QPushButton::clicked, this, &MainWindow::on_import_start_clicked);
  connect(stop_command_button_, &QPushButton::clicked, this, &MainWindow::on_import_stop_clicked);
  connect(record_command_button_, &QPushButton::clicked, this,
          &MainWindow::on_export_record_clicked);
  connect(stop_record_command_button_, &QPushButton::clicked, this,
          &MainWindow::on_export_stop_clicked);

  main_splitter_ = new QSplitter(Qt::Horizontal, central_widget);
  main_splitter_->setChildrenCollapsible(false);
  main_layout->addWidget(main_splitter_, 1);

  auto *controls_content = new QWidget(main_splitter_);
  controls_content->setObjectName("controlsContent");
  controls_content->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
  auto *controls_layout = new QHBoxLayout(controls_content);
  controls_layout->setContentsMargins(0, 0, 0, 0);
  controls_layout->setSpacing(12);

  auto *acquisition_column = new QWidget(controls_content);
  acquisition_column->setObjectName("controlsColumn");
  auto *acquisition_layout = new QVBoxLayout(acquisition_column);
  acquisition_layout->setContentsMargins(0, 0, 0, 0);
  acquisition_layout->setSpacing(12);
  acquisition_layout->addWidget(import_widget_);
  acquisition_layout->addWidget(export_widget_);
  acquisition_layout->addWidget(view_widget_);
  acquisition_layout->addWidget(view_widget_->post_processing_group());
  acquisition_layout->addStretch(1);

  auto *processing_column = new QWidget(controls_content);
  processing_column->setObjectName("controlsColumn");
  auto *processing_layout = new QVBoxLayout(processing_column);
  processing_layout->setContentsMargins(0, 0, 0, 0);
  processing_layout->setSpacing(12);
  processing_layout->addWidget(render_widget_);
  processing_layout->addWidget(render_widget_->autofocus_widget());
  processing_layout->addStretch(1);

  auto *controls_divider = new QFrame(controls_content);
  controls_divider->setObjectName("controlsColumnDivider");
  controls_divider->setFixedWidth(1);
  controls_divider->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

  controls_layout->addWidget(acquisition_column);
  controls_layout->addWidget(controls_divider);
  controls_layout->addWidget(processing_column);

  auto *controls_scroll = new QScrollArea(main_splitter_);
  controls_scroll->setObjectName("controlsScrollArea");
  controls_scroll->viewport()->setObjectName("controlsScrollViewport");
  controls_scroll->setWidgetResizable(true);
  controls_scroll->setFrameShape(QFrame::NoFrame);
  controls_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  controls_scroll->setWidget(controls_content);
  const int controls_width =
      std::max(controls_content->sizeHint().width() +
                   controls_scroll->verticalScrollBar()->sizeHint().width() + 12,
               430);
  controls_scroll->setFixedWidth(controls_width);
  controls_scroll->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

  display_workspace_ = new QWidget(main_splitter_);
  display_workspace_->setObjectName("displayWorkspace");
  auto *display_layout = new QVBoxLayout(display_workspace_);
  display_layout->setContentsMargins(8, 0, 8, 0);
  display_layout->setSpacing(8);

  monitor_widget_ = new SystemMonitorWidget(this);
  monitor_widget_->setObjectName("systemMonitorPanel");
  monitor_widget_->setMinimumWidth(280);

  main_splitter_->addWidget(controls_scroll);
  main_splitter_->addWidget(display_workspace_);
  main_splitter_->addWidget(monitor_widget_);
  main_splitter_->setStretchFactor(0, 0);
  main_splitter_->setStretchFactor(1, 1);
  main_splitter_->setStretchFactor(2, 0);
  main_splitter_->setSizes({controls_width, 780, 300});

  connect(patient_line_edit_, &QLineEdit::textChanged, this,
          &MainWindow::update_recording_path_preview);
  connect(eye_side_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int) { update_recording_path_preview(); });
}

QGroupBox *MainWindow::create_display_panel(const QString &title, const QString &display_id,
                                            TensorDisplayWidget *widget) {
  auto *panel = new DraggableDisplayPanel(
      title, display_id,
      [this](const QString &dropped_id, bool target_main) {
        move_display_panel(dropped_id,
                           target_main ? DisplayPanelZone::Main : DisplayPanelZone::Secondary);
      },
      display_workspace_);

  auto *layout = new QVBoxLayout(panel);
  layout->setContentsMargins(6, 6, 6, 6);
  layout->setSpacing(0);
  layout->addWidget(new SquareDisplayViewport(
      widget, display_id,
      [this](const QString &dropped_id, bool target_main) {
        move_display_panel(dropped_id,
                           target_main ? DisplayPanelZone::Main : DisplayPanelZone::Secondary);
      },
      panel));

  widget->show();
  panel->hide();
  return panel;
}

std::array<QGroupBox *, 9> MainWindow::display_panels() const {
  return {xy_processed_panel_,         xy_raw_panel_,        raw_spectrum_panel_,
          processed_spectrum_panel_,   zernike_phase_panel_, shack_hartmann_panel_,
          shack_hartmann_xcorr_panel_, xz_processed_panel_,  yz_processed_panel_};
}

QGroupBox *MainWindow::display_panel_for(TensorDisplayWidget *widget) const {
  if (widget == xy_processed_widget_) {
    return xy_processed_panel_;
  }
  if (widget == xz_processed_widget_) {
    return xz_processed_panel_;
  }
  if (widget == yz_processed_widget_) {
    return yz_processed_panel_;
  }
  if (widget == xy_raw_widget_) {
    return xy_raw_panel_;
  }
  if (widget == raw_spectrum_widget_) {
    return raw_spectrum_panel_;
  }
  if (widget == processed_spectrum_widget_) {
    return processed_spectrum_panel_;
  }
  if (widget == shack_hartmann_widget_) {
    return shack_hartmann_panel_;
  }
  if (widget == shack_hartmann_xcorr_widget_) {
    return shack_hartmann_xcorr_panel_;
  }
  if (widget == zernike_phase_widget_) {
    return zernike_phase_panel_;
  }

  return nullptr;
}

QGroupBox *MainWindow::display_panel_for_id(const QString &display_id) const {
  for (auto *panel : display_panels()) {
    if (panel != nullptr && panel->property(kDisplayPanelIdProperty).toString() == display_id) {
      return panel;
    }
  }

  return nullptr;
}

void MainWindow::set_display_title(TensorDisplayWidget *widget, const QString &title) {
  if (widget != nullptr) {
    widget->setWindowTitle(title);
  }

  if (auto *panel = display_panel_for(widget); panel != nullptr) {
    panel->setTitle(title);
  }
}

void MainWindow::set_display_visible(TensorDisplayWidget *widget, bool visible) {
  if (widget == nullptr) {
    return;
  }

  if (auto *panel = display_panel_for(widget); panel != nullptr) {
    panel->setVisible(visible);
  }

  widget->setVisible(visible);
  if (display_layout_update_depth_ > 0) {
    display_layout_dirty_ = true;
    return;
  }

  relayout_display_panels();
}

void MainWindow::begin_display_layout_update() { ++display_layout_update_depth_; }

void MainWindow::end_display_layout_update() {
  if (display_layout_update_depth_ <= 0) {
    return;
  }

  --display_layout_update_depth_;
  if (display_layout_update_depth_ == 0 && display_layout_dirty_) {
    display_layout_dirty_ = false;
    relayout_display_panels();
  }
}

void MainWindow::move_display_panel(const QString &display_id, DisplayPanelZone zone) {
  auto *panel = display_panel_for_id(display_id);
  if (panel == nullptr) {
    return;
  }

  set_display_panel_zone(panel, zone);
  relayout_display_panels();
}

void MainWindow::set_display_panel_zone(QGroupBox *panel, DisplayPanelZone zone) {
  if (panel == nullptr) {
    return;
  }

  panel->setProperty(kDisplayPanelMainProperty, zone == DisplayPanelZone::Main);
}

bool MainWindow::is_display_panel_in_main(QGroupBox *panel) const {
  return panel != nullptr && panel->property(kDisplayPanelMainProperty).toBool();
}

QStringList MainWindow::main_display_panel_ids() const {
  QStringList ids;
  for (auto *panel : display_panels()) {
    if (is_display_panel_in_main(panel)) {
      ids << panel->property(kDisplayPanelIdProperty).toString();
    }
  }

  return ids;
}

void MainWindow::relayout_display_panels() {
  if (main_display_layout_ == nullptr || secondary_display_layout_ == nullptr) {
    return;
  }

  clear_layout(main_display_layout_);
  clear_layout(secondary_display_layout_);

  QList<QGroupBox *> main_panels;
  QList<QGroupBox *> secondary_panels;
  for (auto *panel : display_panels()) {
    if (panel == nullptr) {
      continue;
    }

    if (panel->isHidden()) {
      continue;
    }

    if (is_display_panel_in_main(panel)) {
      main_panels << panel;
    } else {
      secondary_panels << panel;
    }
  }

  auto add_panels = [](QGridLayout *layout, const QList<QGroupBox *> &panels, int columns) {
    const int clamped_columns = std::max(1, columns);
    for (int i = 0; i < panels.size(); ++i) {
      layout->addWidget(panels[i], i / clamped_columns, i % clamped_columns);
    }

    for (int column = 0; column < clamped_columns; ++column) {
      layout->setColumnStretch(column, 1);
    }

    const int panel_count = static_cast<int>(panels.size());
    const int rows        = std::max(1, (panel_count + clamped_columns - 1) / clamped_columns);
    for (int row = 0; row < rows; ++row) {
      layout->setRowStretch(row, 1);
    }
  };

  const int main_columns = main_panels.size() <= 1 ? 1 : (main_panels.size() <= 4 ? 2 : 3);
  add_panels(main_display_layout_, main_panels, main_columns);
  add_panels(secondary_display_layout_, secondary_panels, 4);
  refresh_secondary_display_visibility();
}

void MainWindow::refresh_secondary_display_visibility() {
  if (secondary_display_container_ == nullptr) {
    return;
  }

  bool any_secondary_visible = false;
  int  visible_main_count    = 0;
  for (auto *panel : display_panels()) {
    if (panel == nullptr || panel->isHidden()) {
      continue;
    }

    if (is_display_panel_in_main(panel)) {
      ++visible_main_count;
    } else {
      any_secondary_visible = true;
    }
  }

  const bool keep_drop_target_visible = !any_secondary_visible && visible_main_count > 1;
  if (any_secondary_visible) {
    secondary_display_container_->setMinimumHeight(0);
    secondary_display_container_->setMaximumHeight(QWIDGETSIZE_MAX);
  } else if (keep_drop_target_visible) {
    secondary_display_container_->setMinimumHeight(kEmptySecondaryDropHeight);
    secondary_display_container_->setMaximumHeight(kEmptySecondaryDropHeight);
  }

  secondary_display_container_->setVisible(any_secondary_visible || keep_drop_target_visible);
}

QString MainWindow::sanitize_recording_token(const QString &value) const {
  QString token = value.trimmed();
  if (token.isEmpty()) {
    token = "PATIENT";
  }

  token.replace(QRegularExpression(R"(\s+)"), "_");
  token.remove(QRegularExpression(R"([^A-Za-z0-9_-])"));
  token = token.left(64);

  if (token.isEmpty()) {
    token = "PATIENT";
  }

  return token;
}

QString MainWindow::recording_file_name(int acquisition_id) const {
  const QString patient = sanitize_recording_token(patient_line_edit_->text());
  const QString eye     = sanitize_recording_token(eye_side_combo_->currentText());
  return QString("%1_%2_%3_%4.holo")
      .arg(patient, eye, session_id_, acquisition_label(acquisition_id));
}

QString MainWindow::acquisition_label(int acquisition_id) const {
  return QString("AQ%1").arg(std::max(acquisition_id, 1), 3, 10, QChar('0'));
}

void MainWindow::update_acquisition_label() {
  if (acquisition_value_label_ != nullptr) {
    acquisition_value_label_->setText(acquisition_label(next_acquisition_id_));
  }
}

void MainWindow::update_recording_path_preview() {
  if (export_widget_ == nullptr) {
    return;
  }

  QFileInfo info(export_widget_->get_file_path());
  QString   directory = info.absolutePath();
  if (directory.isEmpty() || directory == ".") {
    directory = QDir::currentPath();
  }

  export_widget_->set_file_path(
      QDir(directory).filePath(recording_file_name(next_acquisition_id_)));
  update_acquisition_label();
  refresh_command_bar();
}

void MainWindow::set_status_label(QLabel *label, const QString &text, const char *tone) {
  if (label == nullptr) {
    return;
  }

  label->setText(text);
  label->setProperty("statusTone", QString::fromLatin1(tone));
  label->style()->unpolish(label);
  label->style()->polish(label);
  label->update();
}

void MainWindow::configure_unsupported_features() {
  if (view_widget_ == nullptr || render_widget_ == nullptr) {
    return;
  }

  const QString cuts_tooltip =
      tr("Visible for planned support. The current pipeline does not generate XZ/YZ 3D cuts yet.");
  const QString registration_tooltip =
      tr("Visible for planned support. The current pipeline does not support registration yet.");
  const QString convolution_tooltip =
      tr("Visible for planned support. The current pipeline does not support convolution kernels "
         "yet.");

  {
    QSignalBlocker blocker(view_widget_->cuts_3d_check());
    view_widget_->set_cuts_3d_enabled(false);
  }
  view_widget_->cuts_3d_check()->setEnabled(false);
  view_widget_->cuts_3d_check()->setToolTip(cuts_tooltip);
  view_widget_->x_spin()->setEnabled(false);
  view_widget_->x_width_spin()->setEnabled(false);
  view_widget_->y_spin()->setEnabled(false);
  view_widget_->y_width_spin()->setEnabled(false);
  set_display_visible(xz_processed_widget_, false);
  set_display_visible(yz_processed_widget_, false);

  {
    QSignalBlocker blocker(view_widget_->registration_check());
    view_widget_->set_registration_enabled(false);
  }
  view_widget_->registration_check()->setEnabled(false);
  view_widget_->registration_check()->setToolTip(registration_tooltip);
  view_widget_->registration_radius()->setEnabled(false);
  view_widget_->registration_radius()->setToolTip(registration_tooltip);

  {
    QSignalBlocker blocker(render_widget_->convolution_combo());
    render_widget_->set_convolution(QStringLiteral("None"));
  }
  {
    QSignalBlocker blocker(render_widget_->convolution_divide_check());
    render_widget_->set_convolution_divide(false);
  }
  render_widget_->convolution_combo()->setEnabled(false);
  render_widget_->convolution_combo()->setToolTip(convolution_tooltip);
  render_widget_->convolution_divide_check()->setEnabled(false);
  render_widget_->convolution_divide_check()->setToolTip(convolution_tooltip);
}

void MainWindow::refresh_command_bar() {
  if (start_command_button_ == nullptr || import_widget_ == nullptr || render_widget_ == nullptr ||
      view_widget_ == nullptr || export_widget_ == nullptr) {
    return;
  }

  start_command_button_->setEnabled(import_widget_->start_button()->isEnabled());
  stop_command_button_->setEnabled(import_widget_->stop_button()->isEnabled());
  record_command_button_->setEnabled(export_widget_->record_button()->isEnabled());
  stop_record_command_button_->setEnabled(export_widget_->stop_button()->isEnabled());

  if (pipeline_running_) {
    set_status_label(pipeline_status_label_, "Live", "success");
  } else if (update_in_progress_) {
    set_status_label(pipeline_status_label_, "Updating", "warning");
  } else {
    set_status_label(pipeline_status_label_, "Idle", "muted");
  }

  QString source_text;
  if (import_widget_->is_camera_mode()) {
    source_text = import_widget_->get_camera_type();
  } else {
    const QFileInfo file_info(import_widget_->get_file_path());
    source_text = file_info.fileName().isEmpty() ? QStringLiteral("File") : file_info.fileName();
  }
  source_status_label_->setText(source_text);

  view_status_label_->setText(
      QString("%1 %2").arg(render_widget_->get_image_mode(), view_widget_->get_image_type()));

  if (!pipeline_running_) {
    fps_status_label_->setText("--");
  }

  if (export_in_progress_) {
    set_status_label(recording_status_label_, "Recording", "danger");
  } else if (export_widget_->isChecked()) {
    set_status_label(recording_status_label_, "Armed", "warning");
  } else {
    set_status_label(recording_status_label_, "Off", "muted");
  }
}

void MainWindow::save_persistent_state() {
  QSettings settings;

  settings.beginGroup("main_window");
  settings.setValue("geometry", saveGeometry());
  if (main_splitter_ != nullptr) {
    settings.setValue("splitter_state", main_splitter_->saveState());
  }
  settings.setValue("main_display_panels", main_display_panel_ids());
  settings.endGroup();

  settings.beginGroup("session");
  settings.setValue("patient", patient_line_edit_->text());
  settings.setValue("eye_side", eye_side_combo_->currentText());
  settings.endGroup();

  settings.beginGroup("import");
  settings.setValue("camera_mode", import_widget_->is_camera_mode());
  settings.setValue("file_path", import_widget_->get_file_path());
  settings.setValue("fps_limit", import_widget_->get_fps_limit().value_or(0));
  settings.setValue("start_index", import_widget_->get_start_index());
  settings.setValue("end_index", import_widget_->get_end_index());
  settings.setValue("load_method", import_widget_->get_load_method());
  settings.setValue("camera_type", import_widget_->get_camera_type());
  settings.setValue("camera_config", import_widget_->get_camera_config());
  settings.endGroup();

  settings.beginGroup("rendering");
  settings.setValue("image_mode", render_widget_->get_image_mode());
  settings.setValue("batch_size", render_widget_->get_batch_size());
  settings.setValue("time_stride", render_widget_->get_time_stride());
  settings.setValue("filter_2d", render_widget_->is_filter_2d_enabled());
  settings.setValue("filter_inner", render_widget_->get_filter_inner());
  settings.setValue("filter_outer", render_widget_->get_filter_outer());
  settings.setValue("space_transform", render_widget_->get_space_transform());
  settings.setValue("time_transform", render_widget_->get_time_transform());
  settings.setValue("time_window", render_widget_->get_time_window());
  settings.setValue("lambda_nm", render_widget_->get_lambda());
  settings.setValue("focus_mm", render_widget_->get_focus());
  settings.setValue("convolution", render_widget_->get_convolution());
  settings.setValue("convolution_divide", render_widget_->is_convolution_divide());
  settings.endGroup();

  settings.beginGroup("view");
  settings.setValue("image_type", view_widget_->get_image_type());
  settings.setValue("cuts_3d", view_widget_->is_cuts_3d_enabled());
  settings.setValue("fft_shift", view_widget_->is_fft_shift_enabled());
  settings.setValue("raw_view", view_widget_->is_raw_view_enabled());
  settings.setValue("raw_spectrum", view_widget_->is_raw_spectrum_view_enabled());
  settings.setValue("processed_spectrum", view_widget_->is_process_spectrum_view_enabled());
  settings.setValue("x_origin", view_widget_->get_x_origin());
  settings.setValue("x_width", view_widget_->get_x_width());
  settings.setValue("y_origin", view_widget_->get_y_origin());
  settings.setValue("y_width", view_widget_->get_y_width());
  settings.setValue("z_origin", view_widget_->get_z_origin());
  settings.setValue("z_width", view_widget_->get_z_width());
  settings.setValue("view_kind", view_widget_->get_view_kind());
  settings.setValue("accumulation", view_widget_->get_accumulation());
  settings.setValue("range_start", view_widget_->get_range_start());
  settings.setValue("range_end", view_widget_->get_range_end());
  settings.setValue("registration", view_widget_->is_registration_enabled());
  settings.setValue("registration_radius", view_widget_->get_registration_radius());
  settings.setValue("reticle", view_widget_->is_reticle_enabled());
  settings.setValue("reticle_radius", view_widget_->get_reticle_radius());
  settings.setValue("pct", view_widget_->is_pct_enabled());
  settings.setValue("pct_radius", view_widget_->get_pct_radius());
  settings.endGroup();

  settings.beginGroup("export");
  settings.setValue("enabled", export_widget_->isChecked());
  settings.setValue("image_type", export_widget_->get_image_type());
  settings.setValue("file_path", export_widget_->get_file_path());
  settings.setValue("tag", export_widget_->get_tag());
  settings.setValue("frame_count_enabled", export_widget_->is_frame_count_enabled());
  settings.setValue("frame_count", export_widget_->get_frame_count());
  settings.endGroup();

  auto *autofocus = render_widget_->autofocus_widget();
  settings.beginGroup("autofocus");
  settings.setValue("enabled", autofocus->is_enabled());
  settings.setValue("nb_subaps", autofocus->get_nb_subaps());
  settings.setValue("z2_enabled", autofocus->is_z2_enabled());
  settings.setValue("z3_enabled", autofocus->is_z3_enabled());
  settings.setValue("z4_enabled", autofocus->is_z4_enabled());
  settings.setValue("z5_enabled", autofocus->is_z5_enabled());
  settings.setValue("z6_enabled", autofocus->is_z6_enabled());
  settings.setValue("z7_enabled", autofocus->is_z7_enabled());
  settings.setValue("z8_enabled", autofocus->is_z8_enabled());
  settings.setValue("z9_enabled", autofocus->is_z9_enabled());
  settings.setValue("z10_enabled", autofocus->is_z10_enabled());
  settings.setValue("show_phase", autofocus->show_reconstructed_phase());
  settings.setValue("show_shack_hartmann", autofocus->show_shack_hartmann_sensor_view());
  settings.setValue("show_xcorr", autofocus->show_cross_correlation_view());
  settings.endGroup();

  settings.sync();
}

void MainWindow::restore_persistent_state() {
  QSettings settings;

  settings.beginGroup("main_window");
  if (settings.contains("geometry")) {
    geometry_restored_ = restoreGeometry(settings.value("geometry").toByteArray());
  }
  if (main_splitter_ != nullptr && settings.contains("splitter_state")) {
    main_splitter_->restoreState(settings.value("splitter_state").toByteArray());
  }
  if (settings.contains("main_display_panels")) {
    const QStringList main_panel_ids = settings.value("main_display_panels").toStringList();
    for (auto *panel : display_panels()) {
      set_display_panel_zone(panel, DisplayPanelZone::Secondary);
    }
    for (const auto &display_id : main_panel_ids) {
      if (auto *panel = display_panel_for_id(display_id); panel != nullptr) {
        set_display_panel_zone(panel, DisplayPanelZone::Main);
      }
    }
    relayout_display_panels();
  }
  settings.endGroup();

  settings.beginGroup("session");
  patient_line_edit_->setText(settings.value("patient", patient_line_edit_->text()).toString());
  restore_combo_text(settings, "eye_side", eye_side_combo_);
  settings.endGroup();

  settings.beginGroup("import");
  import_widget_->set_camera_mode(
      settings.value("camera_mode", import_widget_->is_camera_mode()).toBool());
  import_widget_->set_file_path(
      settings.value("file_path", import_widget_->get_file_path()).toString());
  import_widget_->set_fps_limit(settings.value("fps_limit", 0).toInt());
  import_widget_->set_start_index(
      settings.value("start_index", import_widget_->get_start_index()).toInt());
  import_widget_->set_end_index(
      settings.value("end_index", import_widget_->get_end_index()).toInt());
  restore_combo_text(settings, "load_method", import_widget_->load_method_combo());
  restore_combo_text(settings, "camera_type", import_widget_->camera_combo());
  restore_combo_text(settings, "camera_config", import_widget_->camera_config_combo());
  settings.endGroup();

  settings.beginGroup("rendering");
  restore_combo_text(settings, "image_mode", render_widget_->image_combo());
  render_widget_->set_batch_size(
      settings.value("batch_size", render_widget_->get_batch_size()).toInt());
  render_widget_->set_time_stride(
      settings.value("time_stride", render_widget_->get_time_stride()).toInt());
  render_widget_->set_filter_2d_enabled(
      settings.value("filter_2d", render_widget_->is_filter_2d_enabled()).toBool());
  render_widget_->set_filter_inner(
      settings.value("filter_inner", render_widget_->get_filter_inner()).toInt());
  render_widget_->set_filter_outer(
      settings.value("filter_outer", render_widget_->get_filter_outer()).toInt());
  restore_combo_text(settings, "space_transform", render_widget_->space_transform_combo());
  restore_combo_text(settings, "time_transform", render_widget_->time_transform_combo());
  render_widget_->set_time_window(
      settings.value("time_window", render_widget_->get_time_window()).toInt());
  render_widget_->set_lambda(settings.value("lambda_nm", render_widget_->get_lambda()).toInt());
  render_widget_->set_focus(settings.value("focus_mm", render_widget_->get_focus()).toInt());
  restore_combo_text(settings, "convolution", render_widget_->convolution_combo());
  render_widget_->set_convolution_divide(
      settings.value("convolution_divide", render_widget_->is_convolution_divide()).toBool());
  settings.endGroup();

  settings.beginGroup("view");
  restore_combo_text(settings, "image_type", view_widget_->image_type_combo());
  view_widget_->set_cuts_3d_enabled(
      settings.value("cuts_3d", view_widget_->is_cuts_3d_enabled()).toBool());
  view_widget_->set_fft_shift_enabled(
      settings.value("fft_shift", view_widget_->is_fft_shift_enabled()).toBool());
  view_widget_->raw_view_check()->setChecked(
      settings.value("raw_view", view_widget_->is_raw_view_enabled()).toBool());
  view_widget_->raw_spectrum_view_check()->setChecked(
      settings.value("raw_spectrum", view_widget_->is_raw_spectrum_view_enabled()).toBool());
  view_widget_->process_spectrum_view_check()->setChecked(
      settings.value("processed_spectrum", view_widget_->is_process_spectrum_view_enabled())
          .toBool());
  view_widget_->set_x_origin(settings.value("x_origin", view_widget_->get_x_origin()).toInt());
  view_widget_->set_x_width(settings.value("x_width", view_widget_->get_x_width()).toInt());
  view_widget_->set_y_origin(settings.value("y_origin", view_widget_->get_y_origin()).toInt());
  view_widget_->set_y_width(settings.value("y_width", view_widget_->get_y_width()).toInt());
  view_widget_->set_z_origin(settings.value("z_origin", view_widget_->get_z_origin()).toInt());
  view_widget_->set_z_width(settings.value("z_width", view_widget_->get_z_width()).toInt());
  restore_combo_text(settings, "view_kind", view_widget_->kind_combo());
  view_widget_->set_accumulation(
      settings.value("accumulation", view_widget_->get_accumulation()).toInt());
  view_widget_->range_start_spin()->setValue(
      settings.value("range_start", view_widget_->get_range_start()).toInt());
  view_widget_->range_end_spin()->setValue(
      settings.value("range_end", view_widget_->get_range_end()).toInt());
  view_widget_->set_registration_enabled(
      settings.value("registration", view_widget_->is_registration_enabled()).toBool());
  view_widget_->registration_radius()->setValue(
      settings.value("registration_radius", view_widget_->get_registration_radius()).toDouble());
  view_widget_->reticle_check()->setChecked(
      settings.value("reticle", view_widget_->is_reticle_enabled()).toBool());
  view_widget_->reticle_radius()->setValue(
      settings.value("reticle_radius", view_widget_->get_reticle_radius()).toDouble());
  view_widget_->set_pct_enabled(settings.value("pct", view_widget_->is_pct_enabled()).toBool());
  view_widget_->set_pct_radius(
      settings.value("pct_radius", view_widget_->get_pct_radius()).toDouble());
  settings.endGroup();

  settings.beginGroup("export");
  export_widget_->setChecked(settings.value("enabled", export_widget_->isChecked()).toBool());
  restore_combo_text(settings, "image_type", export_widget_->image_type_combo());
  export_widget_->set_file_path(
      settings.value("file_path", export_widget_->get_file_path()).toString());
  restore_combo_text(settings, "tag", export_widget_->tag_combo());
  export_widget_->frames_check()->setChecked(
      settings.value("frame_count_enabled", export_widget_->is_frame_count_enabled()).toBool());
  export_widget_->set_frame_count(
      settings.value("frame_count", export_widget_->get_frame_count()).toInt());
  settings.endGroup();

  auto *autofocus = render_widget_->autofocus_widget();
  settings.beginGroup("autofocus");
  autofocus->nb_subaps_spin()->setValue(
      settings.value("nb_subaps", autofocus->get_nb_subaps()).toInt());
  autofocus->set_z2_enabled(settings.value("z2_enabled", autofocus->is_z2_enabled()).toBool());
  autofocus->set_z3_enabled(settings.value("z3_enabled", autofocus->is_z3_enabled()).toBool());
  autofocus->set_z4_enabled(settings.value("z4_enabled", autofocus->is_z4_enabled()).toBool());
  autofocus->set_z5_enabled(settings.value("z5_enabled", autofocus->is_z5_enabled()).toBool());
  autofocus->set_z6_enabled(settings.value("z6_enabled", autofocus->is_z6_enabled()).toBool());
  autofocus->set_z7_enabled(settings.value("z7_enabled", autofocus->is_z7_enabled()).toBool());
  autofocus->set_z8_enabled(settings.value("z8_enabled", autofocus->is_z8_enabled()).toBool());
  autofocus->set_z9_enabled(settings.value("z9_enabled", autofocus->is_z9_enabled()).toBool());
  autofocus->set_z10_enabled(settings.value("z10_enabled", autofocus->is_z10_enabled()).toBool());
  autofocus->set_show_reconstructed_phase(
      settings.value("show_phase", autofocus->show_reconstructed_phase()).toBool());
  autofocus->set_show_shack_hartmann_sensor_view(
      settings.value("show_shack_hartmann", autofocus->show_shack_hartmann_sensor_view()).toBool());
  autofocus->set_show_cross_correlation_view(
      settings.value("show_xcorr", autofocus->show_cross_correlation_view()).toBool());
  autofocus->set_enabled(settings.value("enabled", autofocus->is_enabled()).toBool());
  settings.endGroup();

  update_recording_path_preview();
}

void MainWindow::initialize_display_widgets() {
  xy_raw_widget_               = new TensorDisplayWidget(display_workspace_);
  xy_processed_widget_         = new TensorDisplayWidget(display_workspace_);
  xz_processed_widget_         = new TensorDisplayWidget(display_workspace_);
  yz_processed_widget_         = new TensorDisplayWidget(display_workspace_);
  raw_spectrum_widget_         = new TensorDisplayWidget(display_workspace_);
  processed_spectrum_widget_   = new TensorDisplayWidget(display_workspace_);
  shack_hartmann_widget_       = new TensorDisplayWidget(display_workspace_);
  shack_hartmann_xcorr_widget_ = new TensorDisplayWidget(display_workspace_);
  zernike_phase_widget_        = new TensorDisplayWidget(display_workspace_);

  set_display_title(xy_raw_widget_, "XY-Raw");
  set_display_title(xy_processed_widget_, "XY-Processed");
  set_display_title(xz_processed_widget_, "XZ-Processed");
  set_display_title(yz_processed_widget_, "YZ-Processed");
  set_display_title(raw_spectrum_widget_, "Raw Spectrum");
  set_display_title(processed_spectrum_widget_, "Processed Spectrum");
  set_display_title(shack_hartmann_widget_, "Shack Hartmann");
  set_display_title(shack_hartmann_xcorr_widget_, "Shack Hartmann XCorr");
  set_display_title(zernike_phase_widget_, "Zernike Phase");

  zernike_phase_widget_->set_colormap(Colormap::Twilight);
  zernike_phase_widget_->set_value_range(0.0f, 2 * static_cast<float>(M_PI));

  auto *display_layout = static_cast<QVBoxLayout *>(display_workspace_->layout());

  main_display_container_ = new DisplayDropZone(
      [this](const QString &display_id) { move_display_panel(display_id, DisplayPanelZone::Main); },
      display_workspace_);
  main_display_container_->setObjectName("mainDisplayZone");
  main_display_layout_ = new QGridLayout(main_display_container_);
  main_display_layout_->setContentsMargins(0, 0, 0, 0);
  main_display_layout_->setSpacing(8);
  display_layout->addWidget(main_display_container_, 3);

  secondary_display_container_ = new DisplayDropZone(
      [this](const QString &display_id) {
        move_display_panel(display_id, DisplayPanelZone::Secondary);
      },
      display_workspace_);
  secondary_display_container_->setObjectName("secondaryDisplayZone");
  secondary_display_layout_ = new QGridLayout(secondary_display_container_);
  secondary_display_layout_->setContentsMargins(0, 0, 0, 0);
  secondary_display_layout_->setSpacing(8);

  xy_processed_panel_ = create_display_panel("XY-Processed", "xy_processed", xy_processed_widget_);
  xy_raw_panel_       = create_display_panel("XY-Raw", "xy_raw", xy_raw_widget_);
  raw_spectrum_panel_ = create_display_panel("Raw Spectrum", "raw_spectrum", raw_spectrum_widget_);
  processed_spectrum_panel_ =
      create_display_panel("Processed Spectrum", "processed_spectrum", processed_spectrum_widget_);
  zernike_phase_panel_ =
      create_display_panel("Zernike Phase", "zernike_phase", zernike_phase_widget_);
  shack_hartmann_panel_ =
      create_display_panel("Shack Hartmann", "shack_hartmann", shack_hartmann_widget_);
  shack_hartmann_xcorr_panel_ = create_display_panel("Shack Hartmann XCorr", "shack_hartmann_xcorr",
                                                     shack_hartmann_xcorr_widget_);
  xz_processed_panel_ = create_display_panel("XZ-Processed", "xz_processed", xz_processed_widget_);
  yz_processed_panel_ = create_display_panel("YZ-Processed", "yz_processed", yz_processed_widget_);

  set_display_panel_zone(xy_processed_panel_, DisplayPanelZone::Main);
  relayout_display_panels();

  display_layout->addWidget(secondary_display_container_, 2);
  begin_display_layout_update();
  set_display_visible(xy_processed_widget_, false);
  set_display_visible(xy_raw_widget_, false);
  set_display_visible(raw_spectrum_widget_, false);
  set_display_visible(processed_spectrum_widget_, false);
  set_display_visible(zernike_phase_widget_, false);
  set_display_visible(shack_hartmann_widget_, false);
  set_display_visible(shack_hartmann_xcorr_widget_, false);
  set_display_visible(xz_processed_widget_, false);
  set_display_visible(yz_processed_widget_, false);
  end_display_layout_update();

  connect(view_widget_, &ViewWidget::cuts_3d_toggled, this, [this](bool checked) {
    if (checked) {
      set_display_visible(xz_processed_widget_, true);
      set_display_visible(yz_processed_widget_, true);
    } else {
      set_display_visible(xz_processed_widget_, false);
      set_display_visible(yz_processed_widget_, false);
    }
  });

  connect(view_widget_, &ViewWidget::raw_view_toggled, this, [this](bool checked) {
    if (pipeline_running_ && checked) {
      set_display_visible(xy_raw_widget_, true);
    } else {
      set_display_visible(xy_raw_widget_, false);
    }
  });

  connect(view_widget_, &ViewWidget::raw_spectrum_view_toggled, this, [this](bool checked) {
    if (pipeline_running_ && checked) {
      set_display_visible(raw_spectrum_widget_, true);
    } else {
      set_display_visible(raw_spectrum_widget_, false);
    }
  });

  connect(view_widget_, &ViewWidget::process_spectrum_view_toggled, this, [this](bool checked) {
    if (pipeline_running_ && checked) {
      set_display_visible(processed_spectrum_widget_, true);
    } else {
      set_display_visible(processed_spectrum_widget_, false);
    }
  });

  connect(view_widget_, &ViewWidget::reticle_toggled, this, [this](bool checked) {
    if (xy_processed_widget_) {
      xy_processed_widget_->set_reticle_enabled(checked);
      if (checked) {
        xy_processed_widget_->set_reticle_radius(view_widget_->get_reticle_radius());
      }
    }
  });

  connect(view_widget_, &ViewWidget::reticle_radius_changed, this, [this](double value) {
    if (xy_processed_widget_ && view_widget_->is_reticle_enabled()) {
      xy_processed_widget_->set_reticle_radius(value);
    }
  });
}

void MainWindow::initialize_pipeline_manager() {
  pipeline_manager_ = new pipeline::Manager(
      render_widget_->autofocus_widget(), xy_processed_widget_, xz_processed_widget_,
      yz_processed_widget_, xy_raw_widget_, raw_spectrum_widget_, processed_spectrum_widget_,
      shack_hartmann_widget_, shack_hartmann_xcorr_widget_, zernike_phase_widget_);
  pipeline_manager_thread_ = new QThread(this);
  pipeline_manager_->moveToThread(pipeline_manager_thread_);
  pipeline_manager_thread_->start();
}

void MainWindow::connect_manager_signals() {
  connect(pipeline_manager_, &pipeline::Manager::start_pipeline_success, this,
          &MainWindow::on_start_pipeline_success);

  connect(pipeline_manager_, &pipeline::Manager::start_pipeline_failure, this,
          &MainWindow::on_start_pipeline_failure);

  connect(pipeline_manager_, &pipeline::Manager::stop_pipeline_success, this,
          &MainWindow::on_stop_pipeline_success);

  connect(pipeline_manager_, &pipeline::Manager::stop_pipeline_failure, this,
          &MainWindow::on_stop_pipeline_failure);

  connect(pipeline_manager_, &pipeline::Manager::update_pipeline_success, this,
          &MainWindow::on_update_pipeline_success);

  connect(pipeline_manager_, &pipeline::Manager::update_pipeline_failure, this,
          &MainWindow::on_update_pipeline_failure);

  connect(pipeline_manager_, &pipeline::Manager::metrics_updated, this,
          &MainWindow::on_metrics_updated, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_started_success, this,
          &MainWindow::on_raw_record_started_success, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_started_failure, this,
          &MainWindow::on_raw_record_started_failure, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_stopped_success, this,
          &MainWindow::on_raw_record_stopped_success, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_stopped_failure, this,
          &MainWindow::on_raw_record_stopped_failure, Qt::QueuedConnection);
}

void MainWindow::connect_import_controls() {
  connect(import_widget_, &ImportWidget::start_clicked, this, &MainWindow::on_import_start_clicked);
  connect(import_widget_, &ImportWidget::stop_clicked, this, &MainWindow::on_import_stop_clicked);
  connect(import_widget_, &ImportWidget::browse_clicked, this, [=]() {
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (file.isEmpty()) {
      return;
    }
    try {
      auto reader      = holofile::Reader(file.toStdString());
      auto frame_count = reader.header().frame_count;

      import_widget_->set_file_path(file);
      import_widget_->set_end_index_range(0, static_cast<int>(frame_count));
      import_widget_->set_end_index(static_cast<int>(frame_count));
      import_widget_->set_start_index(0);

      if (reader.footer().has_value()) {
        auto               footer        = reader.footer().value();
        pipeline::Settings prev_settings = get_pipeline_settings();
        prev_settings.view_type          = pipeline::ViewType::PROCESSED;
        pipeline::Settings new_settings =
            pipeline::old_json_to_settings(footer.pipeline_settings, prev_settings);
        set_pipeline_settings(new_settings);
      }
    } catch (std::exception &e) {
      logger()->error("failed to open \"{}\": \"{}\"", file.toStdString(), e.what());
    }
  });
}

void MainWindow::connect_export_controls() {

  connect(export_widget_, &ExportWidget::record_clicked, this,
          &MainWindow::on_export_record_clicked);
  connect(export_widget_, &ExportWidget::stop_clicked, this, &MainWindow::on_export_stop_clicked);
  connect(export_widget_, &ExportWidget::browse_clicked, this, [=]() {
    QFileInfo current_info(export_widget_->get_file_path());
    QString   initial_directory = current_info.absolutePath();
    if (initial_directory.isEmpty() || initial_directory == ".") {
      initial_directory = QDir::currentPath();
    }

    QString directory = QFileDialog::getExistingDirectory(this, tr("Select Recording Directory"),
                                                          initial_directory);
    if (!directory.isEmpty()) {
      export_widget_->set_file_path(
          QDir(directory).filePath(recording_file_name(next_acquisition_id_)));
    }
  });
}

void MainWindow::configure_window() {
  setWindowTitle("Holovibes");

  const QSize minimum_size(960, 640);
  const QSize default_size = minimum_size.expandedTo(QSize(1280, 800));
  setMinimumSize(minimum_size);
  if (!geometry_restored_) {
    resize(default_size);
  }
}

std::filesystem::path MainWindow::makeRecordingPath(const QString &userText) {
  namespace fs = std::filesystem;

  QFileInfo info(userText);

  fs::path dir = info.absolutePath().isEmpty() ? fs::current_path()
                                               : fs::path(info.absolutePath().toStdString());

  int      acquisition_id = next_acquisition_id_;
  QString  finalFileName  = recording_file_name(acquisition_id);
  fs::path candidate      = dir / finalFileName.toStdString();

  // Add suffix if needed
  while (fs::exists(candidate)) {
    ++acquisition_id;
    finalFileName = recording_file_name(acquisition_id);
    candidate     = dir / finalFileName.toStdString();
  }

  pending_recording_acquisition_id_ = acquisition_id;
  QSignalBlocker blocker(export_widget_->file_line_edit());
  export_widget_->set_file_path(QString::fromStdString(candidate.string()));

  logger()->info("[MainWindow::makeRecordingPath] Generated recording path: {}",
                 candidate.string());

  return candidate;
}

void MainWindow::show_pipeline_error_popup(const QString &message) {
  QMessageBox msgBox(this);
  msgBox.setIcon(QMessageBox::Critical);
  msgBox.setWindowTitle(tr("Pipeline Error"));
  msgBox.setText(message);
  msgBox.setStandardButtons(QMessageBox::Ok);
  msgBox.setDefaultButton(QMessageBox::Ok);
  msgBox.exec();
}

void MainWindow::on_start_pipeline_success() {
  logger()->info("[MainWindow::on_start_pipeline_success]");
  pipeline_running_ = true;
  import_widget_->set_stop_enabled(true);
  export_widget_->set_record_enabled(!export_in_progress_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(export_in_progress_);

  if (render_widget_->get_image_mode() == "Raw") {
    set_display_title(xy_processed_widget_, "XY-Raw");
  } else {
    set_display_title(xy_processed_widget_, "XY-Processed");
  }

  auto dims = guess_source_dims();
  xy_raw_widget_->set_fixed_aspect(dims);
  if (render_widget_->get_space_transform() == "Fresnel Diffraction" &&
      render_widget_->get_image_mode() != "Raw") {
    dims = QSize(dims.width(), dims.width());
  }
  xy_processed_widget_->set_fixed_aspect(dims);
  xy_raw_widget_->set_fixed_aspect(guess_source_dims());
  shack_hartmann_widget_->set_fixed_aspect(dims);
  shack_hartmann_xcorr_widget_->set_fixed_aspect(dims);
  zernike_phase_widget_->set_fixed_aspect(guess_source_dims());

  begin_display_layout_update();
  set_display_visible(xy_processed_widget_, true);

  auto *autofocus_widget = render_widget_->autofocus_widget();
  if (autofocus_widget->is_enabled()) {
    const bool has_enabled_zernike =
        autofocus_widget->is_z2_enabled() || autofocus_widget->is_z3_enabled() ||
        autofocus_widget->is_z4_enabled() || autofocus_widget->is_z5_enabled() ||
        autofocus_widget->is_z6_enabled() || autofocus_widget->is_z7_enabled() ||
        autofocus_widget->is_z8_enabled() || autofocus_widget->is_z9_enabled() ||
        autofocus_widget->is_z10_enabled();
    if (!has_enabled_zernike) {
      autofocus_widget->reset_zernike_values();
    }

    if (autofocus_widget->show_shack_hartmann_sensor_view()) {
      set_display_visible(shack_hartmann_widget_, true);
    } else {
      set_display_visible(shack_hartmann_widget_, false);
    }

    if (autofocus_widget->show_cross_correlation_view()) {
      set_display_visible(shack_hartmann_xcorr_widget_, true);
    } else {
      set_display_visible(shack_hartmann_xcorr_widget_, false);
    }

    if (autofocus_widget->show_reconstructed_phase()) {
      set_display_visible(zernike_phase_widget_, true);
    } else {
      set_display_visible(zernike_phase_widget_, false);
    }
  } else {
    set_display_visible(shack_hartmann_widget_, false);
    set_display_visible(shack_hartmann_xcorr_widget_, false);
    set_display_visible(zernike_phase_widget_, false);
    autofocus_widget->reset_zernike_values();
  }

  if (view_widget_->is_raw_view_enabled()) {
    set_display_visible(xy_raw_widget_, true);
  }

  if (view_widget_->is_raw_spectrum_view_enabled()) {
    set_display_visible(raw_spectrum_widget_, true);
  }

  if (view_widget_->is_process_spectrum_view_enabled()) {
    set_display_visible(processed_spectrum_widget_, true);
  }
  end_display_layout_update();
  refresh_command_bar();
}

void MainWindow::on_start_pipeline_failure(const QString &error) {
  logger()->error("[MainWindow::on_start_pipeline_failure]");
  pipeline_running_   = false;
  export_in_progress_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  refresh_command_bar();

  show_pipeline_error_popup(tr("An error occurred while starting the pipeline:\n%1").arg(error));
}

void MainWindow::on_stop_pipeline_success() {
  logger()->info("[MainWindow::on_stop_pipeline_success]");
  pipeline_running_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_in_progress_ = false;
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  on_metrics_updated(0.0);

  begin_display_layout_update();
  set_display_visible(xy_raw_widget_, false);
  set_display_visible(xy_processed_widget_, false);
  set_display_visible(xz_processed_widget_, false);
  set_display_visible(yz_processed_widget_, false);
  set_display_visible(raw_spectrum_widget_, false);
  set_display_visible(processed_spectrum_widget_, false);
  set_display_visible(shack_hartmann_widget_, false);
  set_display_visible(shack_hartmann_xcorr_widget_, false);
  set_display_visible(zernike_phase_widget_, false);
  end_display_layout_update();
  render_widget_->autofocus_widget()->reset_zernike_values();
  refresh_command_bar();
}

void MainWindow::on_stop_pipeline_failure(const QString &error) {
  logger()->error("[MainWindow::on_stop_pipeline_failure]");
  pipeline_running_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_in_progress_ = false;
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  on_metrics_updated(0.0);

  begin_display_layout_update();
  set_display_visible(xy_raw_widget_, false);
  set_display_visible(xy_processed_widget_, false);
  set_display_visible(xz_processed_widget_, false);
  set_display_visible(yz_processed_widget_, false);
  set_display_visible(raw_spectrum_widget_, false);
  set_display_visible(processed_spectrum_widget_, false);
  set_display_visible(shack_hartmann_widget_, false);
  set_display_visible(shack_hartmann_xcorr_widget_, false);
  set_display_visible(zernike_phase_widget_, false);
  end_display_layout_update();
  render_widget_->autofocus_widget()->reset_zernike_values();
  refresh_command_bar();

  show_pipeline_error_popup(error);
}

void MainWindow::on_metrics_updated(double input_fps) {
  if (input_fps < 0.0) {
    input_fps = 0.0;
  }

  int fps = static_cast<int>(input_fps);

  const QString text = QString("%1 fps").arg(fps, 6, 10, QChar('0'));

  if (fps_status_label_ != nullptr) {
    fps_status_label_->setText(text);
  }
  monitor_widget_->set_input_throughput_fps(text);

  monitor_widget_->set_gpu_load("N/A");
  monitor_widget_->set_cpu_load("N/A");
  monitor_widget_->set_input_throughput_bytes("N/A");
  monitor_widget_->set_cpu_throughput("N/A");
  monitor_widget_->set_gpu_throughput("N/A");
  monitor_widget_->set_ram_usage("N/A");
  monitor_widget_->set_vram_usage("N/A");
  monitor_widget_->set_dropped_frames("N/A");
  monitor_widget_->set_pipeline_latency("N/A");
  refresh_command_bar();
}

void MainWindow::on_update_pipeline_success() {
  logger()->info("[MainWindow::on_update_pipeline_success]");
  update_in_progress_ = false;
  import_widget_->set_start_enabled(false);
  export_widget_->set_record_enabled(!export_in_progress_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(export_in_progress_);

  if (render_widget_->get_image_mode() == "Raw") {
    set_display_title(xy_processed_widget_, "XY-Raw");
  } else {
    set_display_title(xy_processed_widget_, "XY-Processed");
  }

  auto dims = guess_source_dims();
  xy_raw_widget_->set_fixed_aspect(dims);
  if (render_widget_->get_space_transform() == "Fresnel Diffraction" &&
      render_widget_->get_image_mode() != "Raw") {
    dims = QSize(dims.width(), dims.width());
  }
  xy_processed_widget_->set_fixed_aspect(dims);
  begin_display_layout_update();
  if (view_widget_->is_raw_view_enabled()) {
    set_display_visible(xy_raw_widget_, true);
  }

  shack_hartmann_widget_->set_fixed_aspect(dims);
  shack_hartmann_xcorr_widget_->set_fixed_aspect(dims);
  zernike_phase_widget_->set_fixed_aspect(guess_source_dims());

  auto *autofocus_widget = render_widget_->autofocus_widget();
  if (autofocus_widget->is_enabled()) {
    const bool has_enabled_zernike =
        autofocus_widget->is_z2_enabled() || autofocus_widget->is_z3_enabled() ||
        autofocus_widget->is_z4_enabled() || autofocus_widget->is_z5_enabled() ||
        autofocus_widget->is_z6_enabled() || autofocus_widget->is_z7_enabled() ||
        autofocus_widget->is_z8_enabled() || autofocus_widget->is_z9_enabled() ||
        autofocus_widget->is_z10_enabled();
    if (!has_enabled_zernike) {
      autofocus_widget->reset_zernike_values();
    }

    if (autofocus_widget->show_shack_hartmann_sensor_view()) {
      set_display_visible(shack_hartmann_widget_, true);
    } else {
      set_display_visible(shack_hartmann_widget_, false);
    }

    if (autofocus_widget->show_cross_correlation_view()) {
      set_display_visible(shack_hartmann_xcorr_widget_, true);
    } else {
      set_display_visible(shack_hartmann_xcorr_widget_, false);
    }

    if (autofocus_widget->show_reconstructed_phase()) {
      set_display_visible(zernike_phase_widget_, true);
    } else {
      set_display_visible(zernike_phase_widget_, false);
    }
  } else {
    set_display_visible(shack_hartmann_widget_, false);
    set_display_visible(shack_hartmann_xcorr_widget_, false);
    set_display_visible(zernike_phase_widget_, false);
    autofocus_widget->reset_zernike_values();
  }
  end_display_layout_update();
  refresh_command_bar();
}

void MainWindow::on_update_pipeline_failure(const QString &error) {
  logger()->error("[MainWindow::on_update_pipeline_failure]");
  pipeline_running_   = false;
  update_in_progress_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_in_progress_ = false;
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  refresh_command_bar();

  show_pipeline_error_popup(tr("An error occurred while updating the pipeline:\n%1").arg(error));
}

void MainWindow::closeEvent(QCloseEvent *event) {
  save_persistent_state();

  begin_display_layout_update();
  set_display_visible(xy_processed_widget_, false);
  set_display_visible(xz_processed_widget_, false);
  set_display_visible(yz_processed_widget_, false);
  set_display_visible(xy_raw_widget_, false);
  set_display_visible(raw_spectrum_widget_, false);
  set_display_visible(processed_spectrum_widget_, false);
  set_display_visible(shack_hartmann_widget_, false);
  set_display_visible(shack_hartmann_xcorr_widget_, false);
  set_display_visible(zernike_phase_widget_, false);
  end_display_layout_update();

  if (pipeline_manager_ && import_widget_->is_stop_enabled()) {
    pipeline_manager_->stop_pipeline();
  }

  if (pipeline_manager_thread_) {
    pipeline_manager_thread_->quit();
    pipeline_manager_thread_->wait();
  }

  QMainWindow::closeEvent(event);
  QCoreApplication::quit();
}

void MainWindow::on_import_start_clicked() {
  HOLOVIBES_CHECK(!pipeline_running_);

  if (!validate_inputs()) {
    return;
  }

  import_widget_->set_start_enabled(false);
  refresh_command_bar();
  pipeline::Settings settings = get_pipeline_settings();
  auto               update   = [=]() { pipeline_manager_->update_pipeline(settings); };
  auto               start    = [=]() { pipeline_manager_->start_pipeline(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, update, Qt::QueuedConnection));
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, start, Qt::QueuedConnection));
}

void MainWindow::on_import_stop_clicked() {
  HOLOVIBES_CHECK(pipeline_running_);
  import_widget_->set_stop_enabled(false);
  refresh_command_bar();
  auto stop = [=]() { pipeline_manager_->stop_pipeline(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, stop, Qt::QueuedConnection));
}

void MainWindow::on_export_record_clicked() {
  if (export_in_progress_) {
    logger()->warn("[MainWindow::on_export_record_clicked] Recording already in progress");
    return;
  }
  if (!pipeline_running_) {
    logger()->warn("[MainWindow::on_export_record_clicked] Pipeline is not running");
    return;
  }

  if (!validate_inputs()) {
    logger()->warn("[MainWindow::on_export_record_clicked] Validation failed");
    export_widget_->set_stop_enabled(pipeline_running_);
    return;
  }

  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  refresh_command_bar();

  std::filesystem::path record_path = makeRecordingPath(export_widget_->get_file_path());
  std::optional<size_t> frame_count;
  if (export_widget_->is_frame_count_enabled()) {
    frame_count = static_cast<size_t>(export_widget_->get_frame_count());
  }

  auto start = [mgr = pipeline_manager_, record_path]() { mgr->start_raw_record(record_path); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, start, Qt::QueuedConnection));
}

void MainWindow::on_export_stop_clicked() {
  if (!export_in_progress_) {
    logger()->warn("[MainWindow::on_export_stop_clicked] No active recording to stop");
    return;
  }

  export_widget_->set_record_enabled(false);
  refresh_command_bar();
  auto stop = [mgr = pipeline_manager_]() { mgr->stop_raw_record(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, stop, Qt::QueuedConnection));
}

void MainWindow::on_raw_record_started_success() {
  logger()->info("[MainWindow::on_raw_record_started_success]");
  export_in_progress_ = true;
  export_widget_->set_stop_enabled(true);
  export_widget_->set_record_enabled(false);
  refresh_command_bar();
}

void MainWindow::on_raw_record_started_failure(const QString &error) {
  logger()->error("[MainWindow::on_raw_record_started_failure]");
  export_in_progress_ = false;
  pending_recording_acquisition_id_.reset();
  export_widget_->set_record_enabled(pipeline_running_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(false);
  refresh_command_bar();

  show_pipeline_error_popup(tr("An error occurred while starting raw recording:\n%1").arg(error));
}

void MainWindow::on_raw_record_stopped_success() {
  logger()->info("[MainWindow::on_raw_record_stopped_success]");
  export_in_progress_ = false;
  if (pending_recording_acquisition_id_.has_value()) {
    next_acquisition_id_ = std::max(next_acquisition_id_, *pending_recording_acquisition_id_ + 1);
    pending_recording_acquisition_id_.reset();
    update_recording_path_preview();
  }
  export_widget_->set_record_enabled(pipeline_running_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(false);
  refresh_command_bar();
}

void MainWindow::on_raw_record_stopped_failure(const QString &error) {
  logger()->error("[MainWindow::on_raw_record_stopped_failure]");
  export_widget_->set_stop_enabled(pipeline_running_);
  refresh_command_bar();
  show_pipeline_error_popup(tr("An error occurred while stopping raw recording:\n%1").arg(error));
}

bool MainWindow::validate_inputs() {
  import_widget_->clear_validation_styles();
  export_widget_->clear_validation_styles();
  render_widget_->clear_validation_styles();
  view_widget_->clear_validation_styles();
  render_widget_->autofocus_widget()->clear_validation_styles();
  configure_unsupported_features();

  pipeline::Settings settings = get_pipeline_settings();
  const auto result = pipeline::validate_settings(settings, build_validation_context(settings));
  refresh_validation_tooltips(result);
  apply_validation_result(result);
  configure_unsupported_features();
  return result.ok();
}

void MainWindow::apply_validation_result(const pipeline::ValidationResult &result) {
  using pipeline::SettingsField;

  for (const auto &issue : result.issues) {
    for (const auto field : issue.fields) {
      switch (field) {
      case SettingsField::LoadPath:
        import_widget_->mark_file_invalid();
        break;
      case SettingsField::CameraConfigPath:
        import_widget_->mark_camera_config_invalid();
        break;
      case SettingsField::LoadBegin:
        import_widget_->mark_start_index_invalid();
        break;
      case SettingsField::LoadEnd:
        import_widget_->mark_end_index_invalid();
        break;
      case SettingsField::LoadBatch:
        render_widget_->mark_batch_size_invalid();
        break;
      case SettingsField::Filter2D:
        render_widget_->mark_filter_2d_invalid();
        break;
      case SettingsField::Filter2DInnerRadius:
        render_widget_->mark_filter_inner_invalid();
        break;
      case SettingsField::Filter2DOuterRadius:
        render_widget_->mark_filter_outer_invalid();
        break;
      case SettingsField::SpacialMethod:
        render_widget_->mark_space_transform_invalid();
        break;
      case SettingsField::TimeMethod:
        render_widget_->mark_time_transform_invalid();
        break;
      case SettingsField::TimeWindow:
        render_widget_->mark_time_window_invalid();
        break;
      case SettingsField::TimeStride:
        render_widget_->mark_time_stride_invalid();
        break;
      case SettingsField::TimeZBegin:
        view_widget_->mark_z_invalid();
        break;
      case SettingsField::TimeZEnd:
        view_widget_->mark_z_width_invalid();
        break;
      case SettingsField::View3DCuts:
        view_widget_->mark_cuts_3d_invalid();
        break;
      case SettingsField::ViewRawSpectrum:
        view_widget_->mark_raw_spectrum_invalid();
        break;
      case SettingsField::ViewProcessedSpectrum:
        view_widget_->mark_processed_spectrum_invalid();
        break;
      case SettingsField::PpConvolution:
        render_widget_->mark_convolution_invalid();
        break;
      case SettingsField::PpRegistration:
        view_widget_->mark_registration_invalid();
        break;
      case SettingsField::RecordingPath:
        export_widget_->mark_file_invalid();
        break;
      case SettingsField::RecordingCount:
        export_widget_->mark_frames_invalid();
        break;
      case SettingsField::AutofocusNbSubaps:
        render_widget_->autofocus_widget()->mark_nb_subaps_invalid();
        break;
      }
    }
  }
}

void MainWindow::refresh_validation_tooltips(const pipeline::ValidationResult &result) {
  using pipeline::SettingsField;

  const std::array bindings = {
      FieldBinding{SettingsField::LoadPath, import_widget_->file_line_edit()},
      FieldBinding{SettingsField::CameraConfigPath, import_widget_->camera_config_combo()},
      FieldBinding{SettingsField::LoadBegin, import_widget_->start_index_spin()},
      FieldBinding{SettingsField::LoadEnd, import_widget_->end_index_spin()},
      FieldBinding{SettingsField::LoadBatch, render_widget_->batch_size_spin()},
      FieldBinding{SettingsField::Filter2D, render_widget_->filter_2d_check()},
      FieldBinding{SettingsField::Filter2DInnerRadius, render_widget_->filter_2d_inner_spin()},
      FieldBinding{SettingsField::Filter2DOuterRadius, render_widget_->filter_2d_outer_spin()},
      FieldBinding{SettingsField::SpacialMethod, render_widget_->space_transform_combo()},
      FieldBinding{SettingsField::TimeMethod, render_widget_->time_transform_combo()},
      FieldBinding{SettingsField::TimeWindow, render_widget_->time_window_spin()},
      FieldBinding{SettingsField::TimeStride, render_widget_->time_stride_spin()},
      FieldBinding{SettingsField::TimeZBegin, view_widget_->z_spin()},
      FieldBinding{SettingsField::TimeZEnd, view_widget_->z_width_spin()},
      FieldBinding{SettingsField::View3DCuts, view_widget_->cuts_3d_check()},
      FieldBinding{SettingsField::ViewRawSpectrum, view_widget_->raw_spectrum_view_check()},
      FieldBinding{SettingsField::ViewProcessedSpectrum,
                   view_widget_->process_spectrum_view_check()},
      FieldBinding{SettingsField::PpConvolution, render_widget_->convolution_combo()},
      FieldBinding{SettingsField::PpRegistration, view_widget_->registration_check()},
      FieldBinding{SettingsField::RecordingPath, export_widget_->file_line_edit()},
      FieldBinding{SettingsField::RecordingCount, export_widget_->frames_spin()},
      FieldBinding{SettingsField::AutofocusNbSubaps,
                   render_widget_->autofocus_widget()->nb_subaps_spin()},
  };

  for (const auto &binding : bindings) {
    const auto  issues  = result.issues_for(binding.field);
    const auto &help    = pipeline::get_field_help(binding.field);
    const auto  tooltip = build_field_tooltip(help, std::span{issues});
    binding.widget->setToolTip(tooltip);
  }
}

pipeline::ValidationContext
MainWindow::build_validation_context(const pipeline::Settings &settings) const {
  pipeline::ValidationContext context;

  if (settings.import_source == pipeline::ImportSource::HOLOFILE) {
    context.load_path_exists =
        !settings.load_path.empty() && std::filesystem::exists(settings.load_path);

    if (context.load_path_exists) {
      try {
        auto header                = holofile::Reader(settings.load_path.string()).header();
        context.source_width       = static_cast<int>(header.frame_width);
        context.source_height      = static_cast<int>(header.frame_height);
        context.source_frame_count = static_cast<int>(header.frame_count);
      } catch (...) {
        context.load_path_exists = false;
      }
    }
  } else {
    context.camera_config_exists = !settings.camera_config_path.empty() &&
                                   std::filesystem::exists(settings.camera_config_path);

    if (context.camera_config_exists) {
      try {
        std::ifstream cfg_file(settings.camera_config_path);
        auto          cfg_json = nlohmann::json::parse(cfg_file);
        const auto   &cfg = cfg_json.contains("s711") ? cfg_json.at("s711") : cfg_json.at("s710");

        context.source_width        = cfg.at("Width").get<int>();
        context.source_height       = cfg.at("Height").get<int>();
        context.camera_config_valid = true;
      } catch (...) {
        context.camera_config_valid = false;
      }
    }
  }

  if (settings.recording_method != pipeline::RecordingMethod::NONE) {
    context.recording_path_error = pipeline::validate_recording_path(settings.recording_path);
  }

  return context;
}

void MainWindow::setup_validation_connections() {
  bool (MainWindow::*cb)() = &MainWindow::validate_inputs;

  // Connect to widget signals
  connect(import_widget_, &ImportWidget::settings_changed, this, cb);
  connect(export_widget_, &ExportWidget::settings_changed, this, cb);
  connect(render_widget_, &ImageRenderingWidget::settings_changed, this, cb);
  connect(view_widget_, &ViewWidget::settings_changed, this, cb);

  connect(import_widget_, &ImportWidget::settings_changed, this, &MainWindow::refresh_command_bar);
  connect(export_widget_, &ExportWidget::settings_changed, this, &MainWindow::refresh_command_bar);
  connect(render_widget_, &ImageRenderingWidget::settings_changed, this,
          &MainWindow::refresh_command_bar);
  connect(view_widget_, &ViewWidget::settings_changed, this, &MainWindow::refresh_command_bar);
}

void MainWindow::setup_update_connections() {
  void (MainWindow::*cb)() = &MainWindow::update_if_running;

  connect(import_widget_, &ImportWidget::settings_changed, this, cb);
  connect(export_widget_, &ExportWidget::settings_changed, this, cb);
  connect(render_widget_, &ImageRenderingWidget::settings_changed, this, cb);
  connect(view_widget_, &ViewWidget::settings_changed, this, cb);
}

void MainWindow::update_if_running() {
  if (!pipeline_manager_ || !import_widget_->is_stop_enabled()) {
    return;
  }

  if (update_in_progress_) {
    return;
  }

  if (!validate_inputs()) {
    return;
  }

  update_in_progress_ = true;
  import_widget_->set_start_enabled(false);
  refresh_command_bar();
  pipeline::Settings settings = get_pipeline_settings();
  auto               update   = [=]() { pipeline_manager_->update_pipeline(settings); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, update, Qt::QueuedConnection));
}

QSize MainWindow::guess_source_dims() {
  if (!import_widget_->is_camera_mode()) {
    auto header     = holofile::Reader(import_widget_->get_file_path().toStdString()).header();
    int  src_width  = header.frame_width;
    int  src_height = header.frame_height;
    return QSize(src_width, src_height);
  }

  else if (import_widget_->is_camera_mode()) {
    auto path     = get_selected_camera_config_path();
    auto cfg_file = std::ifstream(path);
    if (!cfg_file.is_open()) {
      throw std::runtime_error(std::format("Could not open camera config file: {}", path));
    }

    // FIXME: This needs to be properly handeled for more camera support
    auto cfg_json   = nlohmann::json::parse(cfg_file);
    auto cfg        = cfg_json.contains("s711") ? cfg_json.at("s711") : cfg_json.at("s710");
    int  src_width  = cfg.at("Width").get<int>();
    int  src_height = cfg.at("Height").get<int>();
    return QSize(src_width, src_height);
  }

  HOLOVIBES_UNIMPLEMENTED();
}

pipeline::Settings MainWindow::get_pipeline_settings() {
  using namespace holovibes::pipeline;

  Settings s;

  // Advanced Settings
  {
    s.cpu_in_size  = 4096;
    s.gpu_in_size  = 4096;
    s.cpu_rec_size = 4096;
    s.cpu_out_size = 64;
    s.gpu_out_size = 64;
  }

  // Import Settings
  {
    std::map<std::string, LoadMethod> method_from_str{
        {"Read Live", LoadMethod::READ_LIVE},
        {"Load in CPU RAM", LoadMethod::LOAD_IN_CPU},
        {"Load in GPU RAM", LoadMethod::LOAD_IN_GPU},
    };
    std::map<std::string, ImportSource> source_from_str{
        {"Ametek S710 Euresys Coaxlink Octo", ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO},
        {"Ametek S711 Euresys Coaxlink QSFP+", ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP},
    };

    if (!import_widget_->is_camera_mode()) {
      s.import_source  = ImportSource::HOLOFILE;
      s.load_path      = import_widget_->get_file_path().toStdString();
      s.load_begin     = static_cast<size_t>(import_widget_->get_start_index());
      s.load_end       = static_cast<size_t>(import_widget_->get_end_index());
      s.load_fps_limit = import_widget_->get_fps_limit();
      QString method   = import_widget_->get_load_method();
      s.load_method    = method_from_str.at(method.toStdString());
      if (render_widget_->get_time_transform() == "Principal Component Analysis") {
        s.load_batch = 32;
      } else {
        s.load_batch = render_widget_->get_batch_size();
      }
    } else {
      QString source       = import_widget_->get_camera_type();
      s.import_source      = source_from_str.at(source.toStdString());
      s.camera_config_path = get_selected_camera_config_path();
      s.load_batch         = 1;
      s.load_fps_limit     = std::nullopt;

      std::ifstream cfg_file(s.camera_config_path);
      if (cfg_file.is_open()) {
        try {
          auto cfg_json = nlohmann::json::parse(cfg_file);
          s.load_batch  = cfg_json.contains("s711")
                              ? cfg_json.at("s711").at("BufferPartCount").get<int>()
                              : cfg_json.at("s710").at("BufferPartCount").get<int>();
        } catch (...) {
        }
      }
    }
  }

  // Image Rendering Settings
  {
    s.view_type = render_widget_->get_image_mode() == "Raw" ? ViewType::RAW : ViewType::PROCESSED;

    std::map<std::string, SpacialMethod> method_from_str{
        {"None", SpacialMethod::NONE},
        {"Fresnel Diffraction", SpacialMethod::FRESNEL_DIFFRACTION},
        {"Angular Spectrum", SpacialMethod::ANGULAR_SPECTRUM},
    };
    QString method       = render_widget_->get_space_transform();
    s.spacial_method     = method_from_str.at(method.toStdString());
    s.spacial_lambda     = static_cast<float>(render_widget_->get_lambda()) * 1e-9f;
    s.spacial_z          = static_cast<float>(render_widget_->get_focus()) * 1e-3f;
    s.spacial_pixel_size = 20e-6f; // TODO: get from camera
  }
  {
    s.filter_2d           = render_widget_->is_filter_2d_enabled();
    s.filter_r_inner      = render_widget_->get_filter_inner();
    s.filter_r_outer      = render_widget_->get_filter_outer();
    s.filter_smooth_inner = 0;
    s.filter_smooth_outer = 1;
  }
  {
    std::map<std::string, TimeMethod> method_from_str{
        {"None", TimeMethod::NONE},
        {"Principal Component Analysis", TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS},
        {"RFFT", TimeMethod::RFFT},
        {"FFT", TimeMethod::FFT},
    };
    s.time_window       = render_widget_->get_time_window();
    s.time_stride       = render_widget_->get_time_stride();
    s.time_accumulation = 4; // TODO: expose in UI
    QString method      = render_widget_->get_time_transform();
    s.time_method       = method_from_str.at(method.toStdString());
    s.time_x_begin      = view_widget_->get_x_origin();
    s.time_x_end        = s.time_x_begin + view_widget_->get_x_width();
    s.time_y_begin      = view_widget_->get_y_origin();
    s.time_y_end        = s.time_y_begin + view_widget_->get_y_width();
    s.time_z_begin      = view_widget_->get_z_origin();
    s.time_z_end        = s.time_z_begin + view_widget_->get_z_width();
  }

  // View Settings
  {
    std::map<std::string, MomentType> moment_from_str{
        {"M0", MomentType::M0},
        {"M1", MomentType::M1},
        {"M2", MomentType::M2},
    };
    s.view_3d_cuts            = view_widget_->is_cuts_3d_enabled();
    s.raw_view                = view_widget_->is_raw_view_enabled();
    s.moment_type             = moment_from_str.at(view_widget_->get_image_type().toStdString());
    s.view_raw_spectrum       = true;
    s.view_processed_spectrum = true;
  }

  // Post-processing Settings
  {
    QString     appDataBase = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QString     appDataPath = appDataBase + "/" + QCoreApplication::applicationVersion();
    QString     convolutionsKernelsPath = appDataPath + "/" + "convolution_kernels/";
    std::string kernel_path             = convolutionsKernelsPath.toStdString() +
                              render_widget_->get_convolution().toStdString() + ".json";

    s.pp_fps                 = 60;
    s.pp_fft_shift           = view_widget_->is_fft_shift_enabled();
    s.pp_accumulation        = static_cast<size_t>(view_widget_->get_accumulation());
    s.pp_convolution         = render_widget_->get_convolution() != "None";
    s.pp_convolution_path    = kernel_path;
    s.pp_convolution_divide  = render_widget_->is_convolution_divide();
    s.pp_pctclip             = view_widget_->is_pct_enabled();
    s.pp_pctclip_lower       = 0.02f;
    s.pp_pctclip_upper       = 99.98f;
    s.pp_pctclip_radius      = view_widget_->get_pct_radius();
    s.pp_registration        = view_widget_->is_registration_enabled();
    s.pp_registration_radius = view_widget_->get_registration_radius();
  }

  // Recording Settings
  {
    if (export_widget_->isChecked()) {

      s.recording_method = export_widget_->get_image_type() == "Raw Image"
                               ? RecordingMethod::RAW
                               : RecordingMethod::PROCESSED;
    } else {
      s.recording_method = RecordingMethod::NONE;
    }
    s.recording_path  = export_widget_->get_file_path().toStdString();
    s.recording_count = export_widget_->get_frame_count();
  }

  // Auto-Focus Settings
  {
    s.autofocus_enabled        = render_widget_->autofocus_widget()->is_enabled();
    s.autofocus_nb_subaps      = render_widget_->autofocus_widget()->get_nb_subaps();
    s.autofocus_zernike_orders = std::vector<int>();

    if (render_widget_->autofocus_widget()->is_z2_enabled()) {
      s.autofocus_zernike_orders.push_back(2);
    }

    if (render_widget_->autofocus_widget()->is_z3_enabled()) {
      s.autofocus_zernike_orders.push_back(3);
    }

    if (render_widget_->autofocus_widget()->is_z4_enabled()) {
      s.autofocus_zernike_orders.push_back(4);
    }

    if (render_widget_->autofocus_widget()->is_z5_enabled()) {
      s.autofocus_zernike_orders.push_back(5);
    }

    if (render_widget_->autofocus_widget()->is_z6_enabled()) {
      s.autofocus_zernike_orders.push_back(6);
    }

    if (render_widget_->autofocus_widget()->is_z7_enabled()) {
      s.autofocus_zernike_orders.push_back(7);
    }

    if (render_widget_->autofocus_widget()->is_z8_enabled()) {
      s.autofocus_zernike_orders.push_back(8);
    }

    if (render_widget_->autofocus_widget()->is_z9_enabled()) {
      s.autofocus_zernike_orders.push_back(9);
    }

    if (render_widget_->autofocus_widget()->is_z10_enabled()) {
      s.autofocus_zernike_orders.push_back(10);
    }
  }

  return s;
}

void MainWindow::set_pipeline_settings(const pipeline::Settings &s) {
  using namespace holovibes::pipeline;

  // --- Advanced Settings ---
  {
    // (none exposed in UI)
  }

  // --- Import Settings ---
  {
    // Source: camera or file
    if (s.import_source == ImportSource::HOLOFILE) {
      import_widget_->set_camera_mode(false);
      import_widget_->set_file_path(QString::fromStdString(s.load_path.string()));
      import_widget_->set_fps_limit(s.load_fps_limit);
      import_widget_->set_start_index(static_cast<int>(s.load_begin));
      import_widget_->set_end_index(static_cast<int>(s.load_end));

      QString method;
      switch (s.load_method) {
      case LoadMethod::READ_LIVE:
        method = "Read Live";
        break;
      case LoadMethod::LOAD_IN_CPU:
        method = "Load in CPU RAM";
        break;
      case LoadMethod::LOAD_IN_GPU:
        method = "Load in GPU RAM";
        break;
      default:
        method = "Read Live";
        break;
      }
      import_widget_->set_load_method(method);
    } else {
      import_widget_->set_camera_mode(true);
      import_widget_->set_fps_limit(std::nullopt);

      QString source;
      switch (s.import_source) {
      case ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO:
        source = "Ametek S710 Euresys Coaxlink Octo";
        break;
      case ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP:
        source = "Ametek S711 Euresys Coaxlink QSFP+";
        break;
      default:
        source = "Ametek S710 Euresys Coaxlink Octo";
        break;
      }
      import_widget_->set_camera_type(source);
    }
  }

  // --- Image Rendering Settings ---
  {
    QString method;
    switch (s.spacial_method) {
    case SpacialMethod::FRESNEL_DIFFRACTION:
      method = "Fresnel Diffraction";
      break;
    case SpacialMethod::ANGULAR_SPECTRUM:
      method = "Angular Spectrum";
      break;
    default:
      method = "None";
      break;
    }
    render_widget_->set_batch_size(s.load_batch);
    render_widget_->set_space_transform(method);
    render_widget_->set_lambda(s.spacial_lambda * 1e9); // nm
    render_widget_->set_focus(s.spacial_z * 1e3);       // mm
  }
  {
    render_widget_->set_filter_2d_enabled(s.filter_2d);
    render_widget_->set_filter_inner(s.filter_r_inner);
    render_widget_->set_filter_outer(s.filter_r_outer);
    // smooth_inner/smooth_outer omitted since not exposed in UI
  }
  {
    QString method;
    switch (s.time_method) {
    case TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS:
      method = "Principal Component Analysis";
      break;
    case TimeMethod::RFFT:
      method = "RFFT";
      break;
    case TimeMethod::FFT:
      method = "FFT";
      break;
    default:
      method = "None";
      break;
    }
    render_widget_->set_time_transform(method);
    render_widget_->set_time_window(static_cast<int>(s.time_window));
    render_widget_->set_time_stride(static_cast<int>(s.time_stride));

    view_widget_->set_x_origin(static_cast<int>(s.time_x_begin));
    view_widget_->set_x_width(static_cast<int>(s.time_x_end - s.time_x_begin));
    view_widget_->set_y_origin(static_cast<int>(s.time_y_begin));
    view_widget_->set_y_width(static_cast<int>(s.time_y_end - s.time_y_begin));
    view_widget_->set_z_origin(static_cast<int>(s.time_z_begin));
    view_widget_->set_z_width(static_cast<int>(s.time_z_end - s.time_z_begin));
  }

  // --- View Settings ---
  {
    view_widget_->set_cuts_3d_enabled(s.view_3d_cuts);
  }

  // --- Post-processing Settings ---
  {
    view_widget_->set_fft_shift_enabled(s.pp_fft_shift);
    view_widget_->set_accumulation(static_cast<int>(s.pp_accumulation));
    render_widget_->set_convolution_divide(s.pp_convolution_divide);

    // Select convolution kernel name from path
    QString convName = "None";
    if (s.pp_convolution && !s.pp_convolution_path.empty()) {
      auto fileName = QFileInfo(QString::fromStdString(s.pp_convolution_path)).baseName();
      convName      = fileName;
    }
    render_widget_->set_convolution(convName);

    view_widget_->set_pct_radius(s.pp_pctclip_radius);
    view_widget_->set_pct_enabled(s.pp_pctclip);
    view_widget_->set_registration_enabled(s.pp_registration);
    view_widget_->set_registration_radius(s.pp_registration_radius);
  }

  // --- Recording Settings ---
  {
    export_widget_->set_file_path(QString::fromStdString(s.recording_path.string()));
    export_widget_->set_frame_count(static_cast<int>(s.recording_count));
    // recording_method not exposed (always RAW in get_pipeline_settings)
  }

  configure_unsupported_features();
}

std::string MainWindow::get_selected_camera_config_path() {
  QString     appDataBase      = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QString     appDataPath      = appDataBase + "/" + QCoreApplication::applicationVersion();
  QString     cameraConfigPath = appDataPath + "/" + "camera_configs/";
  std::string config_path =
      cameraConfigPath.toStdString() + import_widget_->get_camera_config().toStdString() + ".json";
  return config_path;
}

} // namespace holovibes::ui
