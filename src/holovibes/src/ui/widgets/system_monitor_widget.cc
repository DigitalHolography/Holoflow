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
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QStyleOptionProgressBar>
#include <QVBoxLayout>
#include <algorithm>

namespace holovibes::ui {

class MultiProgressBar : public QProgressBar {
  Q_OBJECT

public:
  explicit MultiProgressBar(QWidget *parent = nullptr) : QProgressBar(parent) {
    setRange(0, 1);
    setTextVisible(true);
    setMinimumHeight(20);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  }

  void set_progress(int first, int second) {
    first_progress_  = std::clamp(first, 0, maximum());
    second_progress_ = std::clamp(second, 0, maximum());

    update();
  }

  void set_first(int first) {
    first_progress_ = std::clamp(first, 0, maximum());
    update();
  }

  void set_second(int second) {
    second_progress_ = std::clamp(second, 0, maximum());
    update();
  }

protected:
  void paintEvent(QPaintEvent *) override {
    QPainter painter(this);

    QStyleOptionProgressBar opt;
    initStyleOption(&opt);

    // 1. Draw the native Qt progress bar background/frame.
    QStyleOptionProgressBar bg = opt;
    bg.progress                = 0;
    bg.textVisible             = false;

    style()->drawControl(QStyle::CE_ProgressBar, &bg, &painter, this);

    // 2. Draw the two colored sections with rounded corners.
    QRect contents = style()->subElementRect(QStyle::SE_ProgressBarContents, &opt, this);

    if (maximum() > 0 && !contents.isEmpty()) {
      qreal unit = qreal(contents.width()) / maximum();

      int first_width  = qRound(first_progress_ * unit);
      int second_width = qRound(second_progress_ * unit);

      painter.save();
      painter.setRenderHint(QPainter::Antialiasing, true);

      // Clip everything to the rounded progress bar shape.
      QPainterPath clipPath;
      qreal        radius = contents.height() / 2.0;
      clipPath.addRoundedRect(contents, radius, radius);
      painter.setClipPath(clipPath);

      // First progress section.
      painter.fillRect(QRect(contents.left(), contents.top(), first_width, contents.height()),
                       first_color_);

      // Second progress section.
      painter.fillRect(QRect(contents.left(), contents.top(), second_width, contents.height()),
                       second_color_);

      painter.restore();
    }

    // 3. Draw centered value / maximum text.
    painter.setPen(opt.palette.color(QPalette::Text));
    painter.drawText(
        rect(), Qt::AlignCenter,
        QString("%1 acq %2 rec / %3").arg(first_progress_).arg(second_progress_).arg(maximum()));
  }

private:
  static inline QColor first_color_     = QColor("#2DA6A1");
  static inline QColor second_color_    = QColor("#3FB950");
  int                  first_progress_  = 0;
  int                  second_progress_ = 0;
};

SystemMonitorWidget::SystemMonitorWidget(QWidget *parent) : QGroupBox("SYSTEM MONITOR", parent) {
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
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(8);

  // Line metrics
  auto *metrics_layout = new QGridLayout();
  metrics_layout->setContentsMargins(0, 0, 0, 0);
  metrics_layout->setHorizontalSpacing(8);
  metrics_layout->setVerticalSpacing(4);
  metrics_layout->setColumnStretch(1, 1);

  auto add_metric_row = [&](int row, const QString &label, QLabel **value_label,
                            const QString &value) {
    auto *name_label = new QLabel(label, this);
    name_label->setMinimumWidth(0);
    name_label->setWordWrap(true);
    metrics_layout->addWidget(name_label, row, 0);
    *value_label = new QLabel(value, this);
    (*value_label)->setMinimumWidth(0);
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
  auto *queue_group  = new QGroupBox("QUEUES", this);
  auto *queue_layout = new QVBoxLayout(queue_group);
  queue_layout->setContentsMargins(0, 0, 0, 0);
  queue_layout->setSpacing(6);
  auto configure_bar = [&](QProgressBar **bar, const QString &title, int value, int maximum) {
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
  queue_layout->addWidget(new QLabel("Record Queue 0", queue_group));
  record_queue_bar_ = new MultiProgressBar(queue_group);
  queue_layout->addWidget(record_queue_bar_);

  layout->addWidget(queue_group);
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

void SystemMonitorWidget::set_record_queue(int first, int second, int max) {
  record_queue_bar_->setMaximum(max);
  record_queue_bar_->set_progress(first, second);
}

void SystemMonitorWidget::update_record_queue_acquired(int first) {
  record_queue_bar_->set_first(first);
}

void SystemMonitorWidget::update_record_queue_recorded(int second) {
  record_queue_bar_->set_second(second);
}
} // namespace holovibes::ui

#include "system_monitor_widget.moc"