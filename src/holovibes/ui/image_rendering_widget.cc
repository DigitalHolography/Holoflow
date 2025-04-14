#include "holovibes/ui/image_rendering_widget.hh"

#include <QLabel>
#include <QSizePolicy>
#include <QSpacerItem>

namespace holovibes::ui {

ImageRenderingWidget::ImageRenderingWidget(QWidget *parent)
    : QGroupBox(tr("Image Rendering"), parent) {
  setup_ui();
}

void ImageRenderingWidget::setup_ui() {
  // Create a grid layout for this group box
  auto *layout = new QGridLayout(this);

  // Row 0: Image selection
  layout->addWidget(new QLabel(tr("Image:")), 0, 0);
  image_combo_ = new QComboBox(this);
  image_combo_->addItems({"Raw", "Processed"});
  layout->addWidget(image_combo_, 0, 1);
  connect(
      image_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
      [this](int index) { emit image_changed(image_combo_->itemText(index)); });

  // Row 1: Batch Size
  layout->addWidget(new QLabel(tr("Batch Size:")), 1, 0);
  batch_size_spin_ = new QSpinBox(this);
  batch_size_spin_->setRange(1, 1024 * 1024);
  batch_size_spin_->setValue(32);
  layout->addWidget(batch_size_spin_, 1, 1);
  connect(batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) {
            emit batch_size_changed(value);
            validate_settings();
          });

  // Row 2: Time Stride
  layout->addWidget(new QLabel(tr("Time Stride:")), 2, 0);
  time_stride_spin_ = new QSpinBox(this);
  time_stride_spin_->setRange(1, 1024 * 1024);
  time_stride_spin_->setValue(32);
  layout->addWidget(time_stride_spin_, 2, 1);
  connect(time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) {
            emit time_stride_changed(value);
            validate_settings();
          });

  // Row 3: Filter 2D
  filter_2d_check_ = new QCheckBox(tr("Filter 2D"), this);
  filter_2d_check_->setLayoutDirection(Qt::RightToLeft);
  layout->addWidget(filter_2d_check_, 3, 0, 1, 2, Qt::AlignRight);
  connect(filter_2d_check_, &QCheckBox::toggled, this,
          &ImageRenderingWidget::filter_2d_toggled);

  // Row 4: Space Transform
  layout->addWidget(new QLabel(tr("Space Transform:")), 4, 0);
  space_transform_combo_ = new QComboBox(this);
  space_transform_combo_->addItems(
      {"None", "Fresnel Diffraction", "Angular Spectrum"});
  layout->addWidget(space_transform_combo_, 4, 1);
  connect(
      space_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
      this, [this](int index) {
        emit space_transform_changed(space_transform_combo_->itemText(index));
      });

  // Row 5: Time Transform
  layout->addWidget(new QLabel(tr("Time Transform:")), 5, 0);
  time_transform_combo_ = new QComboBox(this);
  time_transform_combo_->addItems(
      {"None", "Short Time Fourier Transform", "Principal Component Analysis"});
  layout->addWidget(time_transform_combo_, 5, 1);
  connect(time_transform_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int index) {
            emit time_transform_changed(time_transform_combo_->itemText(index));
          });

  // Row 6: Time Window
  layout->addWidget(new QLabel(tr("Time Window:")), 6, 0);
  time_window_spin_ = new QSpinBox(this);
  time_window_spin_->setRange(1, 1024 * 1024);
  time_window_spin_->setValue(32);
  layout->addWidget(time_window_spin_, 6, 1);
  connect(time_window_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) { emit time_window_changed(value); });

  // Row 7: Lambda (nm)
  layout->addWidget(new QLabel(tr("λ (nm):")), 7, 0);
  lambda_spin_ = new QSpinBox(this);
  lambda_spin_->setRange(1, 1024 * 1024);
  lambda_spin_->setValue(852);
  layout->addWidget(lambda_spin_, 7, 1);
  connect(lambda_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) { emit lambda_changed(value); });

  // Row 8: Boundary (mm)
  layout->addWidget(new QLabel(tr("Boundary (mm):")), 8, 0);
  boundary_spin_ = new QSpinBox(this);
  boundary_spin_->setRange(1, 1024 * 1024);
  boundary_spin_->setValue(0);
  layout->addWidget(boundary_spin_, 8, 1);
  connect(boundary_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) { emit boundary_changed(value); });

  // Row 9: Focus (mm)
  layout->addWidget(new QLabel(tr("Focus (mm):")), 9, 0);
  focus_spin_ = new QSpinBox(this);
  focus_spin_->setRange(1, 1024 * 1024);
  focus_spin_->setValue(380);
  layout->addWidget(focus_spin_, 9, 1);
  connect(focus_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) {
            if (focus_slider_->value() != value) {
              focus_slider_->blockSignals(true);
              focus_slider_->setValue(value);
              focus_slider_->blockSignals(false);
            }
            emit focus_changed(value);
          });

  // Row 10: Focus Slider
  focus_slider_ = new QSlider(Qt::Horizontal, this);
  focus_slider_->setRange(0, 1000);
  focus_slider_->setValue(380);
  layout->addWidget(focus_slider_, 10, 0, 1, 2);
  connect(focus_slider_, &QSlider::valueChanged, this, [this](int value) {
    if (focus_spin_->value() != value) {
      focus_spin_->blockSignals(true);
      focus_spin_->setValue(value);
      focus_spin_->blockSignals(false);
    }
    emit focus_slider_changed(value);
  });

  // Row 11: Convolution label (occupies full row)
  layout->addWidget(new QLabel(tr("Convolution:")), 11, 0, 1, 2);

  // Row 12: Convolution Combo and Divide Check
  convolution_combo_ = new QComboBox(this);
  convolution_combo_->addItems({"None", "Gaussian", "Laplacian"});
  layout->addWidget(convolution_combo_, 12, 0);
  convolution_check_ = new QCheckBox(tr("Divide"), this);
  layout->addWidget(convolution_check_, 12, 1, 1, 1, Qt::AlignRight);
  connect(convolution_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, [this](int index) {
            emit convolution_changed(convolution_combo_->itemText(index),
                                     convolution_check_->isChecked());
          });
  connect(convolution_check_, &QCheckBox::toggled, this, [this](bool checked) {
    emit convolution_changed(convolution_combo_->currentText(), checked);
  });

  // Row 13: Spacer item to push the UI to the top
  auto *spacer =
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  layout->addItem(spacer, 13, 0, 1, 2);
}

void ImageRenderingWidget::validate_settings() {
  int batch_size = batch_size_spin_->value();
  int time_stride = time_stride_spin_->value();
  if (batch_size > 0 && (time_stride % batch_size != 0)) {
    // Indicate an error by setting a red background on the spin boxes.
    batch_size_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
    time_stride_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
  } else {
    // Clear error indication.
    batch_size_spin_->setStyleSheet("");
    time_stride_spin_->setStyleSheet("");
  }
}

} // namespace holovibes::ui
