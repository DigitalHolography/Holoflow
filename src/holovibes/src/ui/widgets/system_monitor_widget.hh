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

#include <QGroupBox>
#include <QLabel>
#include <QProgressBar>

namespace holovibes::ui {

class SystemMonitorWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit SystemMonitorWidget(QWidget *parent = nullptr);

  // Update methods
  void update_metrics(double input_fps);
  void set_gpu_load(const QString &text);
  void set_cpu_load(const QString &text);
  void set_input_throughput_fps(const QString &text);
  void set_input_throughput_bytes(const QString &text);
  void set_cpu_throughput(const QString &text);
  void set_gpu_throughput(const QString &text);
  void set_ram_usage(const QString &text);
  void set_vram_usage(const QString &text);
  void set_dropped_frames(const QString &text);
  void set_pipeline_latency(const QString &text);

  void set_input_queue(int value, int max) {
    input_queue_bar_->setMaximum(max);
    input_queue_bar_->setValue(value);
  }
  void set_output_queue(int value, int max) {
    output_queue_bar_->setMaximum(max);
    output_queue_bar_->setValue(value);
  }
  void set_record_queue(int value, int max) {
    record_queue_bar_->setMaximum(max);
    record_queue_bar_->setValue(value);
  }

private:
  void setup_ui();

  // Metric labels
  QLabel *gpu_load_value_;
  QLabel *cpu_load_value_;
  QLabel *input_throughput_fps_value_;
  QLabel *input_throughput_bytes_value_;
  QLabel *cpu_throughput_value_;
  QLabel *gpu_throughput_value_;
  QLabel *ram_usage_value_;
  QLabel *vram_usage_value_;
  QLabel *dropped_frames_value_;
  QLabel *pipeline_latency_value_;

  // Queue progress bars
  QProgressBar *input_queue_bar_;
  QProgressBar *output_queue_bar_;
  QProgressBar *record_queue_bar_;
};

} // namespace holovibes::ui