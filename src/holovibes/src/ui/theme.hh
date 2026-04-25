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

#pragma once

#include <QApplication>
#include <QColor>
#include <QPalette>
#include <QString>
#include <QStyleFactory>

namespace holovibes::ui {

inline void apply_dark_clinical_theme(QApplication &app) {
  app.setStyle(QStyleFactory::create("Fusion"));

  QPalette palette;
  palette.setColor(QPalette::Window, QColor("#101215"));
  palette.setColor(QPalette::WindowText, QColor("#E6EAF0"));
  palette.setColor(QPalette::Base, QColor("#171A1F"));
  palette.setColor(QPalette::AlternateBase, QColor("#1F2329"));
  palette.setColor(QPalette::ToolTipBase, QColor("#1F2329"));
  palette.setColor(QPalette::ToolTipText, QColor("#E6EAF0"));
  palette.setColor(QPalette::Text, QColor("#E6EAF0"));
  palette.setColor(QPalette::Button, QColor("#1F2329"));
  palette.setColor(QPalette::ButtonText, QColor("#E6EAF0"));
  palette.setColor(QPalette::BrightText, QColor("#F85149"));
  palette.setColor(QPalette::Highlight, QColor("#2DA6A1"));
  palette.setColor(QPalette::HighlightedText, QColor("#050608"));
  palette.setColor(QPalette::Link, QColor("#58A6FF"));
  palette.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#6F7A86"));
  palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#6F7A86"));
  palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#6F7A86"));
  palette.setColor(QPalette::Disabled, QPalette::Button, QColor("#30363D"));
  palette.setColor(QPalette::Disabled, QPalette::Base, QColor("#171A1F"));
  app.setPalette(palette);

  app.setStyleSheet(QStringLiteral(R"(
QWidget {
  background-color: #101215;
  color: #E6EAF0;
  font-size: 13px;
}

QMainWindow,
QDialog,
QMessageBox {
  background-color: #101215;
}

QMenuBar {
  background-color: #171A1F;
  border-bottom: 1px solid #2C333A;
  padding: 2px 6px;
}

QMenuBar::item {
  background: transparent;
  padding: 5px 10px;
  border-radius: 4px;
}

QMenuBar::item:selected {
  background-color: #1F2329;
}

QMenu {
  background-color: #1F2329;
  border: 1px solid #2C333A;
  padding: 4px;
}

QMenu::item {
  padding: 6px 24px;
  border-radius: 3px;
}

QMenu::item:selected {
  background-color: #1C3B3A;
}

QFrame#sessionBar {
  background-color: #171A1F;
  border: 1px solid #2C333A;
  border-radius: 4px;
}

QFrame#sessionBar QLabel {
  color: #A7B0BA;
  font-size: 12px;
}

QFrame#sessionBar QLabel#sessionValue {
  color: #E6EAF0;
  font-weight: 600;
}

QLineEdit#patientField {
  background-color: #0F1216;
  font-weight: 600;
}

QComboBox#eyeSideField {
  background-color: #0F1216;
  font-weight: 600;
  min-width: 64px;
}

QFrame#commandBar {
  background-color: #171A1F;
  border: 1px solid #2C333A;
  border-radius: 4px;
}

QFrame#commandBar QLabel#commandLabel {
  color: #6F7A86;
  font-size: 12px;
}

QFrame#commandBar QLabel#commandValue {
  color: #E6EAF0;
  font-weight: 600;
  font-size: 12px;
}

QPushButton#primaryCommand {
  background-color: #2DA6A1;
  border-color: #2DA6A1;
  color: #050608;
  font-weight: 600;
}

QPushButton#primaryCommand:hover {
  background-color: #38BDB7;
  border-color: #38BDB7;
}

QPushButton#primaryCommand:disabled {
  color: #6F7A86;
  background-color: #171A1F;
  border-color: #30363D;
}

QPushButton#recordCommand {
  border-color: #DA3633;
  color: #F85149;
  font-weight: 600;
}

QPushButton#recordCommand:hover {
  background-color: #3A1D20;
}

QPushButton#recordCommand:disabled {
  color: #6F7A86;
  background-color: #171A1F;
  border-color: #30363D;
}

QGroupBox {
  background-color: #171A1F;
  border: 1px solid #2C333A;
  border-radius: 6px;
  margin-top: 18px;
  padding: 12px;
}

QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 10px;
  padding: 0 4px;
  color: #E6EAF0;
  font-weight: 600;
}

QGroupBox:disabled,
QGroupBox:disabled::title {
  color: #6F7A86;
  border-color: #30363D;
}

QGroupBox#displayPanel {
  background-color: #050608;
  border: 1px solid #2C333A;
  border-radius: 4px;
  margin-top: 18px;
  padding: 8px;
}

QGroupBox#displayPanel[dragActive="true"] {
  border-color: #2DA6A1;
}

QGroupBox#displayPanel::title {
  color: #A7B0BA;
  font-weight: 600;
}

QGroupBox#displayPanel QOpenGLWidget {
  background-color: #050608;
}

QFrame#displayViewport {
  background-color: #050608;
  border: none;
}

QFrame#displayViewport[dragActive="true"],
QFrame#mainDisplayZone[dragActive="true"],
QFrame#secondaryDisplayZone[dragActive="true"] {
  border: 1px solid #2DA6A1;
}

QLabel {
  background: transparent;
}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {
  background-color: #0F1216;
  color: #E6EAF0;
  border: 1px solid #2C333A;
  border-radius: 4px;
  min-height: 24px;
  padding: 3px 7px;
  selection-background-color: #2DA6A1;
  selection-color: #050608;
}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {
  border-color: #2DA6A1;
}

QLineEdit:read-only {
  color: #A7B0BA;
  background-color: #15191E;
}

QFrame#sessionBar QLineEdit,
QFrame#sessionBar QComboBox {
  min-height: 20px;
  padding: 1px 6px;
}

QLineEdit[validationState="error"],
QSpinBox[validationState="error"],
QDoubleSpinBox[validationState="error"],
QComboBox[validationState="error"] {
  color: #F4D6D2;
  background-color: #241416;
  border-color: #F85149;
}

QLineEdit[validationState="error"]:focus,
QSpinBox[validationState="error"]:focus,
QDoubleSpinBox[validationState="error"]:focus,
QComboBox[validationState="error"]:focus {
  border-color: #FF7B72;
}

QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {
  color: #6F7A86;
  background-color: #171A1F;
  border-color: #30363D;
}

QComboBox::drop-down {
  width: 24px;
  border-left: 1px solid #2C333A;
}

QComboBox QAbstractItemView {
  background-color: #1F2329;
  color: #E6EAF0;
  border: 1px solid #2C333A;
  selection-background-color: #1C3B3A;
  selection-color: #E6EAF0;
}

QPushButton {
  background-color: #1F2329;
  color: #E6EAF0;
  border: 1px solid #3A424B;
  border-radius: 4px;
  min-height: 26px;
  padding: 4px 12px;
}

QPushButton:hover {
  background-color: #252B32;
  border-color: #4A5561;
}

QPushButton:pressed {
  background-color: #1C3B3A;
  border-color: #2DA6A1;
}

QPushButton:disabled {
  color: #6F7A86;
  background-color: #171A1F;
  border-color: #30363D;
}

QFrame#commandBar QPushButton {
  min-height: 20px;
  padding: 1px 8px;
}

QCheckBox {
  background: transparent;
  spacing: 7px;
}

QCheckBox:disabled {
  color: #6F7A86;
}

QCheckBox[validationState="error"] {
  color: #FCA5A5;
}

QCheckBox::indicator {
  width: 14px;
  height: 14px;
  border: 1px solid #3A424B;
  border-radius: 3px;
  background-color: #0F1216;
}

QCheckBox::indicator:checked {
  background-color: #2DA6A1;
  border-color: #2DA6A1;
}

QCheckBox[validationState="error"]::indicator {
  background-color: #241416;
  border-color: #F85149;
}

QCheckBox::indicator:disabled {
  background-color: #171A1F;
  border-color: #30363D;
}

QSlider::groove:horizontal {
  height: 4px;
  border-radius: 2px;
  background-color: #30363D;
}

QSlider::handle:horizontal {
  width: 14px;
  margin: -5px 0;
  border-radius: 7px;
  background-color: #2DA6A1;
}

QProgressBar {
  background-color: #0F1216;
  color: #E6EAF0;
  border: 1px solid #2C333A;
  border-radius: 4px;
  min-height: 18px;
  text-align: center;
}

QProgressBar::chunk {
  background-color: #2DA6A1;
  border-radius: 3px;
}

QToolTip {
  background-color: #1F2329;
  color: #E6EAF0;
  border: 1px solid #2C333A;
  padding: 6px;
}

QScrollBar:vertical,
QScrollBar:horizontal {
  background-color: #101215;
  border: none;
  margin: 0;
}

QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {
  background-color: #30363D;
  border-radius: 4px;
  min-height: 24px;
  min-width: 24px;
}

QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {
  background-color: #3A424B;
}
)"));
}

} // namespace holovibes::ui
