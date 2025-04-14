#include "holovibes/ui/view_widget.hh"

#include <QGridLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QSpacerItem>

namespace holovibes::ui {

ViewWidget::ViewWidget(QWidget *parent) : QGroupBox(tr("View"), parent) {
  setup_ui();
}

void ViewWidget::setup_ui() {
  // Create the main grid layout for the View group box.
  auto *view_layout = new QGridLayout(this);

  // Row 0: Image Type
  view_layout->addWidget(new QLabel(tr("Image Type:")), 0, 0);
  image_type_combo_ = new QComboBox(this);
  image_type_combo_->addItems({"Magnitude", "Phase"});
  view_layout->addWidget(image_type_combo_, 0, 1);
  connect(image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, [this](int index) {
            emit image_type_changed(image_type_combo_->itemText(index));
          });

  // Row 1: 3D Cuts and FFT Shift
  cuts_3d_check_ = new QCheckBox(tr("3D Cuts"), this);
  cuts_3d_check_->setLayoutDirection(Qt::LeftToRight);
  view_layout->addWidget(cuts_3d_check_, 1, 0);
  connect(cuts_3d_check_, &QCheckBox::toggled, this,
          &ViewWidget::cuts_3d_toggled);

  fft_shift_check_ = new QCheckBox(tr("FFT Shift"), this);
  fft_shift_check_->setLayoutDirection(Qt::LeftToRight);
  view_layout->addWidget(fft_shift_check_, 1, 1);
  connect(fft_shift_check_, &QCheckBox::toggled, this,
          &ViewWidget::fft_shift_toggled);

  // Row 2: Lens View and Raw View
  lens_view_check_ = new QCheckBox(tr("Lens View"), this);
  lens_view_check_->setLayoutDirection(Qt::LeftToRight);
  view_layout->addWidget(lens_view_check_, 2, 0);
  connect(lens_view_check_, &QCheckBox::toggled, this,
          &ViewWidget::lens_view_toggled);

  raw_view_check_ = new QCheckBox(tr("Raw View"), this);
  raw_view_check_->setLayoutDirection(Qt::LeftToRight);
  view_layout->addWidget(raw_view_check_, 2, 1);
  connect(raw_view_check_, &QCheckBox::toggled, this,
          &ViewWidget::raw_view_toggled);

  // Row 3: Sub-layout for Z and Width
  auto *sub_grid_layout = new QGridLayout();
  view_layout->addLayout(sub_grid_layout, 3, 0, 1, 2);

  sub_grid_layout->addWidget(new QLabel(tr("Z:")), 0, 0);
  z_spin_ = new QSpinBox(this);
  z_spin_->setRange(0, 1024 * 1024);
  z_spin_->setValue(0);
  sub_grid_layout->addWidget(z_spin_, 0, 1);
  connect(z_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::z_changed);

  sub_grid_layout->addWidget(new QLabel(tr("Width:")), 0, 2);
  width_spin_ = new QSpinBox(this);
  width_spin_->setRange(1, 1024 * 1024);
  width_spin_->setValue(0);
  sub_grid_layout->addWidget(width_spin_, 0, 3);
  connect(width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::width_changed);

  // Row 4: View Kind Combo (XY, XZ, YZ)
  view_kind_combo_ = new QComboBox(this);
  view_kind_combo_->addItems({"XY", "XZ", "YZ"});
  view_layout->addWidget(view_kind_combo_, 4, 0, 1, 2);
  connect(view_kind_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, [this](int index) {
            emit view_kind_changed(view_kind_combo_->itemText(index));
          });

  // Row 5: Output Image Accumulation
  view_layout->addWidget(new QLabel(tr("Output image accumulation:")), 5, 0);
  accumulation_spin_ = new QSpinBox(this);
  accumulation_spin_->setRange(1, 1024 * 1024);
  accumulation_spin_->setValue(1);
  view_layout->addWidget(accumulation_spin_, 5, 1);
  connect(accumulation_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::accumulation_changed);

  // Row 6: Brightness/Contrast group box
  brightness_contrast_box_ = new QGroupBox(tr("Brightness/Contrast"), this);
  brightness_contrast_box_->setCheckable(true);
  brightness_contrast_box_->setChecked(true);
  auto *brightness_layout = new QGridLayout(brightness_contrast_box_);
  brightness_contrast_box_->setLayout(brightness_layout);
  view_layout->addWidget(brightness_contrast_box_, 6, 0, 1, 2);

  // Row 6.0: Auto and Invert checkboxes in brightness section
  auto_check_ = new QCheckBox(tr("Auto"), brightness_contrast_box_);
  auto_check_->setLayoutDirection(Qt::RightToLeft);
  brightness_layout->addWidget(auto_check_, 0, 0);
  connect(auto_check_, &QCheckBox::toggled, this, &ViewWidget::auto_changed);

  invert_check_ = new QCheckBox(tr("Invert"), brightness_contrast_box_);
  invert_check_->setLayoutDirection(Qt::RightToLeft);
  brightness_layout->addWidget(invert_check_, 0, 1);
  connect(invert_check_, &QCheckBox::toggled, this,
          &ViewWidget::invert_changed);

  // Row 6.1: Sub-layout for Range spin boxes
  auto *range_layout = new QGridLayout();
  brightness_layout->addLayout(range_layout, 1, 0, 1, 2);
  range_layout->addWidget(new QLabel(tr("Range:")), 0, 0);

  range_start_spin_ = new QSpinBox(brightness_contrast_box_);
  range_start_spin_->setRange(1, 1024 * 1024);
  range_start_spin_->setValue(0);
  range_layout->addWidget(range_start_spin_, 0, 1);
  connect(range_start_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::range_start_changed);

  range_end_spin_ = new QSpinBox(brightness_contrast_box_);
  range_end_spin_->setRange(1, 1024 * 1024);
  range_end_spin_->setValue(255);
  range_layout->addWidget(range_end_spin_, 0, 2);
  connect(range_end_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::range_end_changed);

  // Row 6.2: Renormalize image levels checkbox
  renormalize_check_ =
      new QCheckBox(tr("Renormalize image levels"), brightness_contrast_box_);
  brightness_layout->addWidget(renormalize_check_, 2, 0, 1, 2);
  connect(renormalize_check_, &QCheckBox::toggled, this,
          &ViewWidget::renormalize_changed);

  // Row 7: Spacer to push all elements upward
  auto *spacer_item =
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  view_layout->addItem(spacer_item, 7, 0, 1, 2);
}

} // namespace holovibes::ui
