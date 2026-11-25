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

#include "ui/widgets/pipeline_queues_widget.hh"

#include <QGridLayout>
#include <QSignalBlocker>

namespace holovibes::ui {
namespace {
QSpinBox *create_spin_box(QWidget *parent, int value) {
  auto *spin = new QSpinBox(parent);
  spin->setRange(1, 4096);
  spin->setValue(value);
  return spin;
}
} // namespace

PipelineQueuesWidget::PipelineQueuesWidget(QWidget *parent) : QGroupBox(tr("PIPELINE QUEUES"), parent) {
  auto *layout = new QGridLayout(this);
  layout->setContentsMargins(6, 6, 6, 6);
  layout->setHorizontalSpacing(6);
  layout->setVerticalSpacing(4);
  layout->setColumnStretch(1, 1);
  int row = 0;
  queue_sizing_auto_check_ = new QCheckBox(tr("Automatically size pipeline queues"), this);
  queue_sizing_auto_check_->setToolTip(
      tr("Choose queue depth from frame size; manual values are maximum depths."));
  layout->addWidget(queue_sizing_auto_check_, row++, 0, 1, 2);
  auto add_queue_spin = [&](const QString &name, QLabel *&label, QSpinBox *&spin, int value) {
    label = new QLabel(name, this);
    layout->addWidget(label, row, 0);
    spin = create_spin_box(this, value);
    layout->addWidget(spin, row++, 1);
  };
  add_queue_spin(tr("GPU input queue:"), gpu_in_queue_label_, gpu_in_queue_spin_, 16);
  add_queue_spin(tr("GPU output queue:"), gpu_out_queue_label_, gpu_out_queue_spin_, 2);
  add_queue_spin(tr("CPU output queue:"), cpu_out_queue_label_, cpu_out_queue_spin_, 4);
  update_queue_size_visibility();
  connect(queue_sizing_auto_check_, &QCheckBox::toggled, this, [this]() {
    update_queue_size_visibility();
    emit settings_changed();
  });
  for (auto *spin : {gpu_in_queue_spin_, gpu_out_queue_spin_, cpu_out_queue_spin_})
    connect(spin, qOverload<int>(&QSpinBox::valueChanged), this,
            &PipelineQueuesWidget::settings_changed);
}

bool PipelineQueuesWidget::is_queue_sizing_auto() const { return queue_sizing_auto_check_->isChecked(); }
int PipelineQueuesWidget::get_gpu_in_queue_size() const { return gpu_in_queue_spin_->value(); }
int PipelineQueuesWidget::get_gpu_out_queue_size() const { return gpu_out_queue_spin_->value(); }
int PipelineQueuesWidget::get_cpu_out_queue_size() const { return cpu_out_queue_spin_->value(); }
void PipelineQueuesWidget::set_queue_sizing_auto(bool enabled) { queue_sizing_auto_check_->setChecked(enabled); }
void PipelineQueuesWidget::set_queue_sizes(int gpu_in, int gpu_out, int cpu_out) {
  const QSignalBlocker block_gpu_in(gpu_in_queue_spin_);
  const QSignalBlocker block_gpu_out(gpu_out_queue_spin_);
  const QSignalBlocker block_cpu_out(cpu_out_queue_spin_);
  gpu_in_queue_spin_->setValue(gpu_in);
  gpu_out_queue_spin_->setValue(gpu_out);
  cpu_out_queue_spin_->setValue(cpu_out);
}

void PipelineQueuesWidget::update_queue_size_visibility() {
  const bool visible = !queue_sizing_auto_check_->isChecked();
  gpu_in_queue_label_->setVisible(visible);
  gpu_in_queue_spin_->setVisible(visible);
  gpu_out_queue_label_->setVisible(visible);
  gpu_out_queue_spin_->setVisible(visible);
  cpu_out_queue_label_->setVisible(visible);
  cpu_out_queue_spin_->setVisible(visible);
}

} // namespace holovibes::ui
