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

#pragma once

#include <QCheckBox>
#include <QGroupBox>
#include <QLabel>
#include <QSpinBox>
#include <array>

namespace holovibes::ui {

class PipelineQueuesWidget : public QGroupBox {
  Q_OBJECT
public:
  explicit PipelineQueuesWidget(QWidget *parent = nullptr);
  bool is_queue_sizing_auto() const;
  void set_queue_sizing_auto(bool enabled);
  int max_vram_percent() const;
  int max_ram_percent() const;
  int target_frames_per_batch() const;
  std::array<int, 6> memory_share_plan() const;
  void set_planner_settings(int max_vram, int max_ram, int target_frames, const std::array<int, 6>& shares);

signals:
  void settings_changed();

private:
  void update_queue_size_visibility();
  QCheckBox *queue_sizing_auto_check_;
  QLabel *target_batch_label_;
  QSpinBox *max_vram_spin_;
  QSpinBox *max_ram_spin_;
  QSpinBox *target_batch_spin_;
  std::array<QSpinBox*, 6> share_spins_{};
  QLabel *share_label_;
  bool valid_shares() const;
};

} // namespace holovibes::ui
