#include "holovibes/ui/export_widget.hh"

#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QSpacerItem>

namespace holovibes::ui {

ExportWidget::ExportWidget(QWidget *parent) : QGroupBox(tr("Export"), parent) {
  setup_ui();
}

void ExportWidget::setup_ui() {
  // Overall group layout
  auto *main_layout = new QGridLayout(this);

  // Row 0: Image type combo box
  image_type_combo_ = new QComboBox(this);
  image_type_combo_->addItems({"Raw Image", "Processed Image"});
  main_layout->addWidget(image_type_combo_, 0, 0, 1, 2);
  connect(image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, [this](int index) {
            emit export_image_type_changed(image_type_combo_->itemText(index));
          });

  // Row 1: File path line edit + browse button
  file_line_edit_ = new QLineEdit(this);
  file_line_edit_->setText(QStringLiteral("holovibes\\capture"));
  file_line_edit_->setReadOnly(true);
  main_layout->addWidget(file_line_edit_, 1, 0);

  file_browse_button_ = new QPushButton(tr("..."), this);
  file_browse_button_->setFixedWidth(30);
  main_layout->addWidget(file_browse_button_, 1, 1);

  connect(file_browse_button_, &QPushButton::clicked, this, [this]() {
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (!file.isEmpty()) {
      file_line_edit_->setText(file);
      emit export_file_selected(file);
    }
  });

  // Row 2: "Tag" label + tag combo
  auto *tag_label = new QLabel(tr("Tag"), this);
  main_layout->addWidget(tag_label, 2, 0);

  tag_combo_ = new QComboBox(this);
  tag_combo_->addItems({"None", "Left Eye", "Right Eye"});
  main_layout->addWidget(tag_combo_, 2, 1);
  connect(tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int index) {
            emit export_tag_changed(tag_combo_->itemText(index));
          });

  // Row 3: "Nb. of frames" checkbox + spin box
  frames_check_ = new QCheckBox(tr("Nb. of frames"), this);
  frames_check_->setChecked(true); // or false, depending on your default
  main_layout->addWidget(frames_check_, 3, 0);
  connect(frames_check_, &QCheckBox::toggled, this,
          &ExportWidget::export_frames_check_changed);

  frames_spin_ = new QSpinBox(this);
  frames_spin_->setRange(1, 999999);
  frames_spin_->setValue(2048);
  main_layout->addWidget(frames_spin_, 3, 1);
  connect(frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ExportWidget::export_frames_value_changed);

  // Row 4: Record / Stop / Stop fan buttons
  record_button_ = new QPushButton(tr("Record"), this);
  stop_button_ = new QPushButton(tr("Stop"), this);
  stop_fan_button_ = new QPushButton(tr("Stop fan"), this);

  auto *button_layout = new QHBoxLayout();
  button_layout->addWidget(record_button_);
  button_layout->addWidget(stop_button_);
  button_layout->addWidget(stop_fan_button_);
  main_layout->addLayout(button_layout, 4, 0, 1, 2);

  connect(record_button_, &QPushButton::clicked, this,
          &ExportWidget::export_record_pressed);
  connect(stop_button_, &QPushButton::clicked, this,
          &ExportWidget::export_stop_pressed);
  connect(stop_fan_button_, &QPushButton::clicked, this,
          &ExportWidget::export_stop_fan_pressed);

  // Row 5: Spacer to push everything up
  auto *spacer_item =
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  main_layout->addItem(spacer_item, 5, 0, 1, 2);
}

} // namespace holovibes::ui