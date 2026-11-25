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

namespace holovibes::ui {

class PipelineQueuesWidget : public QGroupBox {
  Q_OBJECT
public:
  explicit PipelineQueuesWidget(QWidget *parent = nullptr);
  bool is_queue_sizing_auto() const;
  int get_gpu_in_queue_size() const;
  int get_gpu_out_queue_size() const;
  int get_cpu_out_queue_size() const;
  void set_queue_sizing_auto(bool enabled);
  void set_queue_sizes(int gpu_in, int gpu_out, int cpu_out);

signals:
  void settings_changed();

private:
  void update_queue_size_visibility();
  QCheckBox *queue_sizing_auto_check_;
  QLabel *gpu_in_queue_label_;
  QLabel *gpu_out_queue_label_;
  QLabel *cpu_out_queue_label_;
  QSpinBox *gpu_in_queue_spin_;
  QSpinBox *gpu_out_queue_spin_;
  QSpinBox *cpu_out_queue_spin_;
};

} // namespace holovibes::ui
