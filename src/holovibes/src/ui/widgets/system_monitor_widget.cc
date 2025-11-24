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

#include "ui/widgets/system_monitor_widget.hh"

#include <QGridLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QVBoxLayout>

namespace holovibes::ui {

SystemMonitorWidget::SystemMonitorWidget(QWidget *parent) : QGroupBox("System Monitor", parent) {
  setup_ui();
}

void SystemMonitorWidget::set_gpu_load(const QString &text) { gpu_load_value_->setText(text); }
void SystemMonitorWidget::set_cpu_load(const QString &text) { cpu_load_value_->setText(text); }
void SystemMonitorWidget::set_input_throughput_fps(const QString &text) {
  input_throughput_fps_value_->setText(text);
}
void SystemMonitorWidget::set_input_throughput_bytes(const QString &text) {
  input_throughput_bytes_value_->setText(text);
}
void SystemMonitorWidget::set_cpu_throughput(const QString &text) {
  cpu_throughput_value_->setText(text);
}
void SystemMonitorWidget::set_gpu_throughput(const QString &text) {
  gpu_throughput_value_->setText(text);
}
void SystemMonitorWidget::set_ram_usage(const QString &text) { ram_usage_value_->setText(text); }
void SystemMonitorWidget::set_vram_usage(const QString &text) { vram_usage_value_->setText(text); }
void SystemMonitorWidget::set_dropped_frames(const QString &text) {
  dropped_frames_value_->setText(text);
}
void SystemMonitorWidget::set_pipeline_latency(const QString &text) {
  pipeline_latency_value_->setText(text);
}

void SystemMonitorWidget::setup_ui() {
  auto *layout = new QVBoxLayout(this);

  // Line metrics
  auto *metrics_layout = new QGridLayout();
  metrics_layout->setColumnStretch(1, 1);

  auto add_metric_row = [&](int row, const QString &label, QLabel **value_label,
                            const QString &value) {
    metrics_layout->addWidget(new QLabel(label, this), row, 0);
    *value_label = new QLabel(value, this);
    (*value_label)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    metrics_layout->addWidget(*value_label, row, 1);
  };

  add_metric_row(0, "GPU Load:", &gpu_load_value_, "N/A");
  add_metric_row(1, "CPU Load:", &cpu_load_value_, "N/A");
  add_metric_row(2, "Input Throughput (FPS):", &input_throughput_fps_value_, "N/A");
  add_metric_row(3, "Input Throughput (Bytes):", &input_throughput_bytes_value_, "N/A");
  add_metric_row(4, "CPU Throughput:", &cpu_throughput_value_, "N/A");
  add_metric_row(5, "GPU Throughput:", &gpu_throughput_value_, "N/A");
  add_metric_row(6, "RAM Usage:", &ram_usage_value_, "N/A");
  add_metric_row(7, "VRAM Usage:", &vram_usage_value_, "N/A");
  add_metric_row(8, "Dropped Frames:", &dropped_frames_value_, "N/A");
  add_metric_row(9, "Pipeline Latency:", &pipeline_latency_value_, "N/A");

  layout->addLayout(metrics_layout);

  // Queue metrics
  auto *queue_group   = new QGroupBox("Queues", this);
  auto *queue_layout  = new QVBoxLayout(queue_group);
  auto  configure_bar = [&](QProgressBar **bar, const QString &title, int value, int maximum) {
    queue_layout->addWidget(new QLabel(title, queue_group));
    *bar = new QProgressBar(queue_group);
    (*bar)->setRange(0, maximum);
    (*bar)->setValue(value);
    (*bar)->setFormat("%v / %m");
    (*bar)->setTextVisible(true);
    queue_layout->addWidget(*bar);
  };

  configure_bar(&input_queue_bar_, "Input Queue", 48, 64);
  configure_bar(&output_queue_bar_, "Output Queue", 22, 64);
  configure_bar(&record_queue_bar_, "Record Queue", 12, 32);

  layout->addWidget(queue_group);
  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding));
  layout->addStretch(1);
}

void SystemMonitorWidget::update_metrics(double input_fps) {
  if (input_fps < 0.0) {
    input_fps = 0.0;
  }

  const QString text = QStringLiteral("%1 fps").arg(QString::number(input_fps, 'f', 1));
  input_throughput_fps_value_->setText(text);

  gpu_load_value_->setText("N/A");
  cpu_load_value_->setText("N/A");
  input_throughput_bytes_value_->setText("N/A");
  cpu_throughput_value_->setText("N/A");
  gpu_throughput_value_->setText("N/A");
  ram_usage_value_->setText("N/A");
  vram_usage_value_->setText("N/A");
  dropped_frames_value_->setText("N/A");
  pipeline_latency_value_->setText("N/A");
}

} // namespace holovibes::ui