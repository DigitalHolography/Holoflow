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
      qreal        radius = contents.height() / 4.0;
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

void RecordMonitorWidget::set_record_size(int max) {
  for (auto &q : record_queues_) {
    q.progress_bar->setMaximum(max);
    q.progress_bar->update();
  }
}

void RecordMonitorWidget::free(size_t q, bool f) {
  assert(q < record_queues_.size());
  auto label = record_queues_[q].state_label;
  if (f) {
    label->setText(QString("FREE"));
    label->setProperty("statusTone", QString::fromLatin1("success"));
    record_queues_[q].progress_bar->set_progress(0, 0);
    record_queues_[q].elapsed_time_label->setText(QString("N/A"));
  } else {
    label->setText(QString("RECORDING"));
    label->setProperty("statusTone", QString::fromLatin1("warning"));
  }

  label->style()->unpolish(label);
  label->style()->polish(label);
  label->update();
}

void RecordMonitorWidget::reset(size_t queue_count) {
  clear_ui();
  setup_ui(queue_count);
}

void RecordMonitorWidget::clear_ui() {
  QLayout *layout = this->layout();
  if (!layout)
    return;

  while (QLayoutItem *item = layout->takeAt(0)) {
    if (QWidget *widget = item->widget())
      delete widget;

    if (QLayout *childLayout = item->layout())
      delete childLayout;

    delete item;
  }

  delete layout;
}

void RecordMonitorWidget::setup_ui(size_t queue_count) {
  // Queue metrics
  auto *queue_layout = new QGridLayout(this);

  auto configure_bar = [&](const QString &title, MultiProgressBar **bar, QLabel **name_label,
                           QLabel **state_label, QLabel **elapsed_time_label, QLabel **speed_label,
                           int &row) {
    *name_label = new QLabel(title, this);
    (*name_label)->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    queue_layout->addWidget(*name_label, row, 0);

    *state_label = new QLabel("FREE", this);
    (*state_label)->setProperty("statusTone", QString::fromLatin1("success"));
    (*state_label)
        ->setToolTip(
            tr("This represent the current state of the record. FREE means that the queue is "
               "available for a record. RECORDING means that the record is not finished yet"));
    (*state_label)->style()->unpolish(*state_label);
    (*state_label)->style()->polish(*state_label);
    (*state_label)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    queue_layout->addWidget(*state_label, row, 1);
    ++row;

    queue_layout->addWidget(new QLabel("Elapsed time: ", this), row, 0);
    *elapsed_time_label = new QLabel("N/A", this);
    (*elapsed_time_label)
        ->setToolTip(tr("Elapsed Time represent the time elapsed since the writing of the record "
                        "on the disk started."));
    (*elapsed_time_label)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    queue_layout->addWidget(*elapsed_time_label, row, 1);
    ++row;

    queue_layout->addWidget(new QLabel("Writing speed: ", this), row, 0);
    *speed_label = new QLabel("N/A", this);
    (*speed_label)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    queue_layout->addWidget(*speed_label, row, 1);
    ++row;

    *bar = new MultiProgressBar(this);
    (*bar)->setRange(0, 1);
    (*bar)->setToolTip(tr(
        "This progress bar shows the progress of the record. The blue part represents the frames "
        "acquired by the camera for this record and the green one the frames written to disk"));
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

    configure_bar(QString("Record queue %1").arg(i), &queue.progress_bar, &queue.name,
                  &queue.state_label, &queue.elapsed_time_label, &queue.writing_speed_value, row);
  }
}

void RecordMonitorWidget::set_record_queue(size_t q, int first, int second, int max) {
  if (q < record_queues_.size()) {
    record_queues_[q].progress_bar->setMaximum(max);
    record_queues_[q].progress_bar->set_progress(first, second);
  }
}

void RecordMonitorWidget::update_record_writing_speed(size_t q, size_t speed) {
  if (q < record_queues_.size()) {
    auto str = QString("%1 frames/s").arg(QString::number(speed));
    record_queues_[q].writing_speed_value->setText(str);
  }
}

void RecordMonitorWidget::update_record_elapsed_time(size_t q, size_t elapsed_time) {
  if (q < record_queues_.size()) {
    auto str = QString("%1 s").arg(QString::number(elapsed_time));
    record_queues_[q].elapsed_time_label->setText(str);
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
