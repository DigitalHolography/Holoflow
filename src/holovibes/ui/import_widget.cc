#include "holovibes/ui/import_widget.hh"

#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QSpacerItem>

#include "holofile/holofile.hh"
#include "holovibes/holovibes.hh"

namespace holovibes::ui {

ImportWidget::ImportWidget(QWidget *parent) : QGroupBox(tr("Import"), parent) {
  setup_ui();
}

void ImportWidget::setup_ui() {
  // Top-level layout
  auto *main_layout = new QGridLayout(this);

  // Row 0: "Select File" line edit + browse button
  file_line_edit_ = new QLineEdit(this);
  file_line_edit_->setPlaceholderText(tr("Select File"));
  file_line_edit_->setReadOnly(true);
  main_layout->addWidget(file_line_edit_, 0, 0);

  file_browse_button_ = new QPushButton(tr("..."), this);
  file_browse_button_->setFixedWidth(30);
  main_layout->addWidget(file_browse_button_, 0, 1);

  connect(file_browse_button_, &QPushButton::clicked, this, [this]() {
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (file.isEmpty()) {
      return;
    }

    auto reader_result = dh::HolofileReader::open(file.toStdString());
    if (!reader_result) {
      dh::holovibes_logger()->error("failed to open \"{}\": \"{}\"",
                                    file.toStdString(),
                                    reader_result.error().message());
      return;
    }
    auto reader = std::move(reader_result.value());
    auto frame_count = reader.header().frame_count;

    file_line_edit_->setText(file);
    start_index_spin_->setValue(0);
    end_index_spin_->setValue(frame_count);
    emit file_selected(file);
  });

  // Row 1: Input FPS
  auto *fps_label = new QLabel(tr("Input FPS"), this);
  main_layout->addWidget(fps_label, 1, 0);

  fps_spin_ = new QSpinBox(this);
  fps_spin_->setRange(1, 999999);
  fps_spin_->setValue(22115);
  main_layout->addWidget(fps_spin_, 1, 1);
  connect(fps_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImportWidget::fps_changed);

  // Row 2: Start Index
  auto *start_index_label = new QLabel(tr("Start Index"), this);
  main_layout->addWidget(start_index_label, 2, 0);

  start_index_spin_ = new QSpinBox(this);
  start_index_spin_->setRange(0, 999999);
  start_index_spin_->setValue(1);
  main_layout->addWidget(start_index_spin_, 2, 1);
  connect(start_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImportWidget::start_index_changed);

  // Row 3: End Index
  auto *end_index_label = new QLabel(tr("End Index"), this);
  main_layout->addWidget(end_index_label, 3, 0);

  end_index_spin_ = new QSpinBox(this);
  end_index_spin_->setRange(1, 999999);
  end_index_spin_->setValue(60);
  main_layout->addWidget(end_index_spin_, 3, 1);
  connect(end_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImportWidget::end_index_changed);

  // Row 4: Load method combo box
  load_method_combo_ = new QComboBox(this);
  load_method_combo_->addItems(
      {"Read Live", "Load in CPU RAM", "Load in GPU RAM"});
  main_layout->addWidget(load_method_combo_, 4, 0, 1, 2);
  connect(load_method_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, [this](int index) {
            emit load_method_changed(load_method_combo_->itemText(index));
          });

  // Row 5: Start / Stop buttons
  start_button_ = new QPushButton(tr("Start"), this);
  stop_button_ = new QPushButton(tr("Stop"), this);
  stop_button_->setEnabled(false);
  auto *button_layout = new QHBoxLayout();
  button_layout->addWidget(start_button_);
  button_layout->addWidget(stop_button_);
  main_layout->addLayout(button_layout, 5, 0, 1, 2);
  connect(start_button_, &QPushButton::clicked, this,
          &ImportWidget::start_import);
  connect(stop_button_, &QPushButton::clicked, this,
          &ImportWidget::stop_import);

  // Row 6: Spacer to push all elements upward
  auto *spacer_item =
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  main_layout->addItem(spacer_item, 6, 0, 1, 2);
}

} // namespace holovibes::ui
