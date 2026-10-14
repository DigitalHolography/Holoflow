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

#include "record_monitor_widget.hh"

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

RecordMonitorWidget::RecordMonitorWidget(QWidget *parent, size_t queue_count)
    : QGroupBox("RECORD MONITOR", parent) {
  setup_ui(queue_count);
}

void RecordMonitorWidget::setup_ui(size_t queue_count) {
  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(8);

  // Queue metrics
  auto *queue_group  = new QGroupBox("QUEUES", this);
  auto *queue_layout = new QGridLayout(queue_group);

  auto configure_bar = [&](const QString &title, MultiProgressBar **bar, QLabel **name_label,
                           QLabel **speed_label, int &row) {
    *name_label = new QLabel(title, queue_group);
    (*name_label)->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    queue_layout->addWidget(*name_label, row, 0);
    ++row;

    queue_layout->addWidget(new QLabel("Writing Speed: ", queue_group), row, 0);
    *speed_label = new QLabel("N/A", queue_group);
    (*speed_label)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    queue_layout->addWidget(*speed_label, row, 1);
    ++row;

    *bar = new MultiProgressBar(queue_group);
    (*bar)->setRange(0, 1);
    (*bar)->setValue(0);
    (*bar)->setTextVisible(true);
    (*bar)->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
    queue_layout->addWidget(*bar, row, 0, 1, 2);
    ++row;
  };

  int row = 0;
  record_queues_.resize(queue_count);
  for (size_t i = 0; i < record_queues_.size(); ++i) {
    RecordQueue &queue = record_queues_[i];

    configure_bar(QString("Record Queue %1").arg(i), &queue.progress_bar, &queue.name,
                  &queue.writing_speed_value, row);
  }

  layout->addWidget(queue_group);
}

void RecordMonitorWidget::set_record_queue(size_t q, int first, int second, int max) {
  if (q < record_queues_.size()) {
    record_queues_[q].progress_bar->setMaximum(max);
    record_queues_[q].progress_bar->set_progress(first, second);
  }
}

void RecordMonitorWidget::update_record_writing_speed(size_t q, size_t first) {
  if (q < record_queues_.size()) {
    auto str = QString("%1 frames/s").arg(QString::number(first));
    record_queues_[q].writing_speed_value->setText(str);
  }
}

void RecordMonitorWidget::update_record_queue_acquired(size_t q, int first) {
  if (q < record_queues_.size()) {
    record_queues_[q].progress_bar->set_first(first);
  }
}

void RecordMonitorWidget::update_record_queue_recorded(size_t q, int second) {
  if (q < record_queues_.size()) {
    record_queues_[q].progress_bar->set_second(second);
  }
}

} // namespace holovibes::ui

#include "record_monitor_widget.moc"
