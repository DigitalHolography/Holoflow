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
#include <vector>

namespace holovibes::ui {

class MultiProgressBar;

class RecordMonitorWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit RecordMonitorWidget(QWidget *parent = nullptr, size_t queue_count = 3);

  // Update methods
  void set_record_size(int max);
  void set_record_queue(size_t q, int first, int second, int max);

  void update_record_queue_acquired(size_t q, int first);
  void update_record_queue_recorded(size_t q, int second);

  void update_record_writing_speed(size_t q, size_t speed);
  void update_record_elapsed_time(size_t q, size_t elapsed_time);

  void free(size_t q, bool f);
private:
  void setup_ui(size_t queue_count);
  struct RecordQueue {
    QLabel           *name;
    QLabel           *writing_speed_value;
    QLabel           *state_label;
    QLabel           *elapsed_time_label;
    MultiProgressBar *progress_bar;
  };

  std::vector<RecordQueue> record_queues_;
};

} // namespace holovibes::ui