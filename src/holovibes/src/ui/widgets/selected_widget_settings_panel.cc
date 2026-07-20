// Copyright 2026 Digital Holography Foundation
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

#include "selected_widget_settings_panel.hh"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QVBoxLayout>

#include <cmath>

#include "ui/widgets/validation_style.hh"
#include "ui/widgets/zernike_history_widget.hh"

namespace holovibes::ui {

class ZernikeHistorySettingsWidget : public QWidget {
public:
  explicit ZernikeHistorySettingsWidget(QWidget *parent = nullptr) : QWidget(parent) {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(2, 2, 2, 2);
    layout->setSpacing(7);

    auto *widget_title = new QLabel("Zernike a4", this);
    widget_title->setObjectName("settingsWidgetTitle");
    layout->addWidget(widget_title);

    auto add_section_label = [this, layout](const QString &text) {
      auto *label = new QLabel(text, this);
      label->setObjectName("settingsSectionLabel");
      layout->addSpacing(5);
      layout->addWidget(label);
    };
    auto add_field = [layout](QLabel *label, QWidget *field) {
      layout->addWidget(label);
      field->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      layout->addWidget(field);
    };

    add_section_label("TIME AXIS");
    auto *time_window_label = new QLabel("Time window", this);
    time_window_spin_       = new QDoubleSpinBox(this);
    time_window_spin_->setRange(0.01, 3600.0);
    time_window_spin_->setDecimals(3);
    time_window_spin_->setSingleStep(0.5);
    time_window_spin_->setSuffix(" s");
    add_field(time_window_label, time_window_spin_);

    add_section_label("Y AXIS");
    auto *scaling_mode_label = new QLabel("Scaling mode", this);
    scaling_mode_combo_      = new QComboBox(this);
    scaling_mode_combo_->addItem("Visible window",
                                 static_cast<int>(YAxisScalingMode::VisibleWindow));
    scaling_mode_combo_->addItem("Recorded extrema",
                                 static_cast<int>(YAxisScalingMode::RecordedExtrema));
    scaling_mode_combo_->addItem("Manual", static_cast<int>(YAxisScalingMode::Manual));
    add_field(scaling_mode_label, scaling_mode_combo_);

    manual_minimum_label_ = new QLabel("Manual minimum", this);
    manual_minimum_spin_  = make_y_spin_box();
    add_field(manual_minimum_label_, manual_minimum_spin_);

    manual_maximum_label_ = new QLabel("Manual maximum", this);
    manual_maximum_spin_  = make_y_spin_box();
    add_field(manual_maximum_label_, manual_maximum_spin_);

    auto *reset_range_button = new QPushButton("Reset range state", this);
    reset_display_button_    = new QPushButton("Reset display settings", this);
    layout->addSpacing(6);
    layout->addWidget(reset_range_button);
    layout->addWidget(reset_display_button_);
    layout->addStretch(1);

    connect(time_window_spin_, &QDoubleSpinBox::editingFinished, this, [this]() {
      if (!target_.isNull()) {
        target_->set_time_window_seconds(time_window_spin_->value());
      }
    });
    connect(scaling_mode_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int index) {
              if (target_.isNull() || index < 0) {
                return;
              }
              target_->set_y_axis_scaling_mode(
                  static_cast<YAxisScalingMode>(scaling_mode_combo_->itemData(index).toInt()));
            });
    connect(manual_minimum_spin_, &QDoubleSpinBox::editingFinished, this,
            [this]() { apply_manual_range(); });
    connect(manual_maximum_spin_, &QDoubleSpinBox::editingFinished, this,
            [this]() { apply_manual_range(); });
    connect(reset_range_button, &QPushButton::clicked, this, [this]() {
      if (!target_.isNull()) {
        target_->reset_recorded_range_state();
      }
    });
    connect(reset_display_button_, &QPushButton::clicked, this, [this]() {
      if (!target_.isNull()) {
        target_->reset_display_settings();
      }
    });
  }

  void set_target(ZernikeHistoryWidget *target) {
    if (settings_connection_) {
      disconnect(settings_connection_);
    }
    target_ = target;
    if (target_.isNull()) {
      return;
    }

    settings_connection_ =
        connect(target_.data(), &ZernikeHistoryWidget::display_settings_changed, this,
                [this](const ZernikeHistoryDisplaySettings &) { synchronize(); });
    synchronize();
  }

private:
  QDoubleSpinBox *make_y_spin_box() {
    auto *spin = new QDoubleSpinBox(this);
    spin->setRange(-1e9, 1e9);
    spin->setDecimals(6);
    spin->setSingleStep(0.1);
    return spin;
  }

  void synchronize() {
    if (target_.isNull()) {
      return;
    }

    const auto           settings = target_->display_settings();
    const QSignalBlocker time_blocker(time_window_spin_);
    const QSignalBlocker mode_blocker(scaling_mode_combo_);
    const QSignalBlocker minimum_blocker(manual_minimum_spin_);
    const QSignalBlocker maximum_blocker(manual_maximum_spin_);

    time_window_spin_->setValue(settings.time_window_seconds);
    const int mode_index = scaling_mode_combo_->findData(static_cast<int>(settings.y_scaling_mode));
    scaling_mode_combo_->setCurrentIndex(mode_index);
    manual_minimum_spin_->setValue(settings.manual_y_minimum);
    manual_maximum_spin_->setValue(settings.manual_y_maximum);
    set_manual_fields_enabled(settings.y_scaling_mode == YAxisScalingMode::Manual);
    clear_validation_error(manual_minimum_spin_);
    clear_validation_error(manual_maximum_spin_);
  }

  void set_manual_fields_enabled(bool enabled) {
    manual_minimum_label_->setEnabled(enabled);
    manual_minimum_spin_->setEnabled(enabled);
    manual_maximum_label_->setEnabled(enabled);
    manual_maximum_spin_->setEnabled(enabled);
  }

  void apply_manual_range() {
    if (target_.isNull()) {
      return;
    }

    const double minimum = manual_minimum_spin_->value();
    const double maximum = manual_maximum_spin_->value();
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum >= maximum) {
      mark_validation_error(manual_minimum_spin_);
      mark_validation_error(manual_maximum_spin_);
      return;
    }

    clear_validation_error(manual_minimum_spin_);
    clear_validation_error(manual_maximum_spin_);
    target_->set_manual_y_range(minimum, maximum);
  }

  QPointer<ZernikeHistoryWidget> target_;
  QMetaObject::Connection        settings_connection_;
  QDoubleSpinBox                *time_window_spin_     = nullptr;
  QComboBox                     *scaling_mode_combo_   = nullptr;
  QLabel                        *manual_minimum_label_ = nullptr;
  QDoubleSpinBox                *manual_minimum_spin_  = nullptr;
  QLabel                        *manual_maximum_label_ = nullptr;
  QDoubleSpinBox                *manual_maximum_spin_  = nullptr;
  QPushButton                   *reset_display_button_ = nullptr;
};

SelectedWidgetSettingsPanel::SelectedWidgetSettingsPanel(QWidget *parent)
    : QGroupBox("WIDGET SETTINGS", parent) {
  setObjectName("selectedWidgetSettingsPanel");
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);

  pages_ = new QStackedWidget(this);

  empty_page_        = new QWidget(pages_);
  auto *empty_layout = new QVBoxLayout(empty_page_);
  empty_layout->setContentsMargins(8, 12, 8, 8);
  auto *placeholder = new QLabel("Click a visualization to configure it", empty_page_);
  placeholder->setObjectName("settingsPlaceholder");
  placeholder->setWordWrap(true);
  placeholder->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
  empty_layout->addWidget(placeholder);
  empty_layout->addStretch(1);
  pages_->addWidget(empty_page_);

  auto *scroll_area = new QScrollArea(pages_);
  scroll_area->setObjectName("widgetSettingsScrollArea");
  scroll_area->setFrameShape(QFrame::NoFrame);
  scroll_area->setWidgetResizable(true);
  scroll_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scroll_area->setMinimumHeight(0);
  scroll_area->viewport()->setObjectName("widgetSettingsScrollViewport");
  zernike_history_page_ = new ZernikeHistorySettingsWidget(scroll_area);
  zernike_history_page_->setObjectName("zernikeHistorySettingsPage");
  scroll_area->setWidget(zernike_history_page_);
  pages_->addWidget(scroll_area);

  layout->addWidget(pages_);
  pages_->setCurrentWidget(empty_page_);
}

void SelectedWidgetSettingsPanel::set_selected_widget(ZernikeHistoryWidget *widget) {
  if (selected_widget_ == widget) {
    zernike_history_page_->set_target(widget);
    return;
  }

  if (destroyed_connection_) {
    disconnect(destroyed_connection_);
  }
  selected_widget_ = widget;
  if (selected_widget_.isNull()) {
    clear_selection();
    return;
  }

  destroyed_connection_ =
      connect(selected_widget_.data(), &QObject::destroyed, this, [this]() { clear_selection(); });
  zernike_history_page_->set_target(selected_widget_.data());
  pages_->setCurrentIndex(1);
}

void SelectedWidgetSettingsPanel::clear_selection() {
  if (destroyed_connection_) {
    disconnect(destroyed_connection_);
    destroyed_connection_ = {};
  }
  selected_widget_.clear();
  zernike_history_page_->set_target(nullptr);
  pages_->setCurrentWidget(empty_page_);
}

} // namespace holovibes::ui
