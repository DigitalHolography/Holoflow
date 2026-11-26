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

#include "ui/widgets/validation_style.hh"

#include <QGridLayout>
#include <QSignalBlocker>
#include <numeric>

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
  auto add_setting = [&](const QString& name, QSpinBox*& spin, int lo, int hi, int val, const QString& tip) {
    auto *label = new QLabel(name, this); layout->addWidget(label, row, 0);
    spin = new QSpinBox(this); spin->setRange(lo, hi); spin->setValue(val); spin->setToolTip(tip);
    label->setToolTip(tip); layout->addWidget(spin, row++, 1);
    connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, &PipelineQueuesWidget::settings_changed);
  };
  add_setting(tr("Maximum VRAM:"), max_vram_spin_, 0, 100, 90, tr("Percent of total VRAM available to queues after subtracting current use."));
  add_setting(tr("Maximum RAM:"), max_ram_spin_, 0, 100, 75, tr("Percent of total host RAM available to queues after current use and 256 MiB minimum headroom."));
  const QString batch_tip = tr("Auto (0) derives frames per batch from frame size to approach ~5 GB/s input throughput.");
  target_batch_label_ = new QLabel(tr("Target frames per batch:"), this);
  layout->addWidget(target_batch_label_, row, 0);
  target_batch_spin_ = new QSpinBox(this);
  target_batch_spin_->setRange(0, 4096);
  target_batch_spin_->setValue(0);
  target_batch_spin_->setSpecialValueText(tr("Auto"));
  target_batch_spin_->setToolTip(batch_tip);
  target_batch_label_->setToolTip(batch_tip);
  layout->addWidget(target_batch_spin_, row++, 1);
  connect(target_batch_spin_, qOverload<int>(&QSpinBox::valueChanged), this, &PipelineQueuesWidget::settings_changed);
  share_label_ = new QLabel(tr("Queue shares (%): input / time-frequency / timing / average / display / recording"), this);
  share_label_->setToolTip(tr("Percentages split the full memory budget among queue groups. If the sum is below 100%, the remainder is shared equally; totals above 100% are invalid."));
  layout->addWidget(share_label_, row++, 0, 1, 2);
  const std::array<int, 6> defaults{35,15,15,15,10,10};
  for (std::size_t i=0; i<share_spins_.size(); ++i) {
    share_spins_[i] = new QSpinBox(this); share_spins_[i]->setRange(0,100); share_spins_[i]->setValue(defaults[i]);
    share_spins_[i]->setToolTip(share_label_->toolTip()); layout->addWidget(share_spins_[i], row, static_cast<int>(i%2));
    if (i%2) ++row;
    connect(share_spins_[i], qOverload<int>(&QSpinBox::valueChanged), this, [this]() { update_queue_size_visibility(); emit settings_changed(); });
  }
  if (share_spins_.size()%2) ++row;
  update_queue_size_visibility();
  connect(queue_sizing_auto_check_, &QCheckBox::toggled, this, [this]() {
    update_queue_size_visibility();
    emit settings_changed();
  });
}

bool PipelineQueuesWidget::is_queue_sizing_auto() const { return queue_sizing_auto_check_->isChecked(); }
void PipelineQueuesWidget::set_queue_sizing_auto(bool enabled) { queue_sizing_auto_check_->setChecked(enabled); }

int PipelineQueuesWidget::max_vram_percent() const { return max_vram_spin_->value(); }
int PipelineQueuesWidget::max_ram_percent() const { return max_ram_spin_->value(); }
int PipelineQueuesWidget::target_frames_per_batch() const { return target_batch_spin_->value(); }
std::array<int, 6> PipelineQueuesWidget::memory_share_plan() const {
  std::array<int, 6> values{}; for (std::size_t i=0;i<values.size();++i) values[i]=share_spins_[i]->value(); return values;
}
void PipelineQueuesWidget::set_planner_settings(int vram,int ram,int target,const std::array<int,6>& shares) {
  max_vram_spin_->setValue(vram); max_ram_spin_->setValue(ram); target_batch_spin_->setValue(is_queue_sizing_auto() ? 0 : target);
  for (std::size_t i=0;i<shares.size();++i) share_spins_[i]->setValue(shares[i]);
  update_queue_size_visibility();
}
bool PipelineQueuesWidget::valid_shares() const { auto values=memory_share_plan(); return std::accumulate(values.begin(), values.end(), 0)<=100; }

void PipelineQueuesWidget::update_queue_size_visibility() {
  const bool visible = !queue_sizing_auto_check_->isChecked();
  target_batch_label_->setVisible(visible);
  target_batch_spin_->setVisible(visible);
  if (!visible && target_batch_spin_->value() != 0) {
    const QSignalBlocker blocker(target_batch_spin_);
    target_batch_spin_->setValue(0);
  }
  share_label_->setVisible(visible);
  const bool shares_valid = valid_shares();
  for (auto *spin : share_spins_) {
    spin->setVisible(visible);
    if (shares_valid) {
      clear_validation_error(spin);
    } else {
      mark_validation_error(spin);
    }
  }
}

} // namespace holovibes::ui
