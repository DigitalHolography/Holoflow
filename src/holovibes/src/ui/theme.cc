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

#include "ui/theme.hh"

#include <QApplication>
#include <QColor>
#include <QFile>
#include <QIODevice>
#include <QPalette>
#include <QString>
#include <QStyleFactory>
#include <QtLogging>

namespace holovibes::ui {

namespace {

struct ClinicalDarkThemeColors {
  QString window           = QStringLiteral("#101215");
  QString panel            = QStringLiteral("#13181E");
  QString panel_dark       = QStringLiteral("#11161B");
  QString surface          = QStringLiteral("#171A1F");
  QString surface_alt      = QStringLiteral("#1F2329");
  QString field            = QStringLiteral("#0F1216");
  QString field_readonly   = QStringLiteral("#15191E");
  QString display          = QStringLiteral("#050608");
  QString border           = QStringLiteral("#2C333A");
  QString border_strong    = QStringLiteral("#3A424B");
  QString border_hover     = QStringLiteral("#4A5561");
  QString button_hover     = QStringLiteral("#252B32");
  QString disabled         = QStringLiteral("#6F7A86");
  QString text             = QStringLiteral("#E6EAF0");
  QString text_muted       = QStringLiteral("#A7B0BA");
  QString text_setting     = QStringLiteral("#A4ACB6");
  QString text_header      = QStringLiteral("#D8DEE6");
  QString accent           = QStringLiteral("#2DA6A1");
  QString accent_hover     = QStringLiteral("#38BDB7");
  QString accent_dark      = QStringLiteral("#1C3B3A");
  QString status_success   = QStringLiteral("#3FB950");
  QString status_warning   = QStringLiteral("#D29922");
  QString danger           = QStringLiteral("#F85149");
  QString danger_border    = QStringLiteral("#DA3633");
  QString danger_bg        = QStringLiteral("#3A1D20");
  QString error_text       = QStringLiteral("#F4D6D2");
  QString error_bg         = QStringLiteral("#241416");
  QString error_focus      = QStringLiteral("#FF7B72");
  QString error_check      = QStringLiteral("#FCA5A5");
  QString link             = QStringLiteral("#58A6FF");
  QString scroll           = QStringLiteral("#30363D");
  QString highlighted_text = QStringLiteral("#050608");
};

struct ClinicalLightThemeColors {
  QString window           = QStringLiteral("#F5F7FA");
  QString panel            = QStringLiteral("#FFFFFF");
  QString panel_dark       = QStringLiteral("#E9EEF3");
  QString surface          = QStringLiteral("#FFFFFF");
  QString surface_alt      = QStringLiteral("#EEF2F6");
  QString field            = QStringLiteral("#FFFFFF");
  QString field_readonly   = QStringLiteral("#F1F4F7");
  QString display          = QStringLiteral("#20252B");
  QString border           = QStringLiteral("#CBD3DC");
  QString border_strong    = QStringLiteral("#AEB9C5");
  QString border_hover     = QStringLiteral("#8795A3");
  QString button_hover     = QStringLiteral("#E2E8EE");
  QString disabled         = QStringLiteral("#8793A0");
  QString text             = QStringLiteral("#20252B");
  QString text_muted       = QStringLiteral("#5D6976");
  QString text_setting     = QStringLiteral("#4F5B68");
  QString text_header      = QStringLiteral("#283440");
  QString accent           = QStringLiteral("#168A86");
  QString accent_hover     = QStringLiteral("#0F706D");
  QString accent_dark      = QStringLiteral("#D5EFED");
  QString status_success   = QStringLiteral("#218739");
  QString status_warning   = QStringLiteral("#A56600");
  QString danger           = QStringLiteral("#C9362B");
  QString danger_border    = QStringLiteral("#B52B22");
  QString danger_bg        = QStringLiteral("#FBE9E7");
  QString error_text       = QStringLiteral("#8F261F");
  QString error_bg         = QStringLiteral("#FDEDEC");
  QString error_focus      = QStringLiteral("#D6453B");
  QString error_check      = QStringLiteral("#E77870");
  QString link             = QStringLiteral("#1268B3");
  QString scroll           = QStringLiteral("#C3CCD5");
  QString highlighted_text = QStringLiteral("#FFFFFF");
};

void set_palette_color(QPalette &palette, QPalette::ColorRole role, const QString &color) {
  palette.setColor(role, QColor(color));
}

void set_disabled_palette_color(QPalette &palette, QPalette::ColorRole role, const QString &color) {
  palette.setColor(QPalette::Disabled, role, QColor(color));
}

template <typename ThemeColors> QPalette build_palette(const ThemeColors &colors) {
  QPalette palette;
  set_palette_color(palette, QPalette::Window, colors.window);
  set_palette_color(palette, QPalette::WindowText, colors.text);
  set_palette_color(palette, QPalette::Base, colors.surface);
  set_palette_color(palette, QPalette::AlternateBase, colors.surface_alt);
  set_palette_color(palette, QPalette::ToolTipBase, colors.surface_alt);
  set_palette_color(palette, QPalette::ToolTipText, colors.text);
  set_palette_color(palette, QPalette::Text, colors.text);
  set_palette_color(palette, QPalette::Button, colors.surface_alt);
  set_palette_color(palette, QPalette::ButtonText, colors.text);
  set_palette_color(palette, QPalette::BrightText, colors.danger);
  set_palette_color(palette, QPalette::Highlight, colors.accent);
  set_palette_color(palette, QPalette::HighlightedText, colors.highlighted_text);
  set_palette_color(palette, QPalette::Link, colors.link);
  set_disabled_palette_color(palette, QPalette::WindowText, colors.disabled);
  set_disabled_palette_color(palette, QPalette::Text, colors.disabled);
  set_disabled_palette_color(palette, QPalette::ButtonText, colors.disabled);
  set_disabled_palette_color(palette, QPalette::Button, colors.scroll);
  set_disabled_palette_color(palette, QPalette::Base, colors.surface);
  return palette;
}

void replace_token(QString &style_sheet, const char *token, const QString &value) {
  style_sheet.replace(QStringLiteral("@%1@").arg(QString::fromLatin1(token)), value);
}

template <typename ThemeColors> QString load_clinical_qss(const ThemeColors &colors) {
  QFile file(QStringLiteral(":/resources/holovibes/styles/clinical_dark.qss"));
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    qWarning() << "Unable to load clinical theme stylesheet:" << file.errorString();
    return {};
  }

  QString style_sheet = QString::fromUtf8(file.readAll());
  replace_token(style_sheet, "WINDOW", colors.window);
  replace_token(style_sheet, "PANEL", colors.panel);
  replace_token(style_sheet, "PANEL_DARK", colors.panel_dark);
  replace_token(style_sheet, "SURFACE", colors.surface);
  replace_token(style_sheet, "SURFACE_ALT", colors.surface_alt);
  replace_token(style_sheet, "FIELD", colors.field);
  replace_token(style_sheet, "FIELD_READONLY", colors.field_readonly);
  replace_token(style_sheet, "DISPLAY", colors.display);
  replace_token(style_sheet, "BORDER", colors.border);
  replace_token(style_sheet, "BORDER_STRONG", colors.border_strong);
  replace_token(style_sheet, "BORDER_HOVER", colors.border_hover);
  replace_token(style_sheet, "BUTTON_HOVER", colors.button_hover);
  replace_token(style_sheet, "DISABLED", colors.disabled);
  replace_token(style_sheet, "TEXT", colors.text);
  replace_token(style_sheet, "TEXT_MUTED", colors.text_muted);
  replace_token(style_sheet, "TEXT_SETTING", colors.text_setting);
  replace_token(style_sheet, "TEXT_HEADER", colors.text_header);
  replace_token(style_sheet, "ACCENT", colors.accent);
  replace_token(style_sheet, "ACCENT_HOVER", colors.accent_hover);
  replace_token(style_sheet, "ACCENT_DARK", colors.accent_dark);
  replace_token(style_sheet, "STATUS_SUCCESS", colors.status_success);
  replace_token(style_sheet, "STATUS_WARNING", colors.status_warning);
  replace_token(style_sheet, "DANGER", colors.danger);
  replace_token(style_sheet, "DANGER_BORDER", colors.danger_border);
  replace_token(style_sheet, "DANGER_BG", colors.danger_bg);
  replace_token(style_sheet, "ERROR_TEXT", colors.error_text);
  replace_token(style_sheet, "ERROR_BG", colors.error_bg);
  replace_token(style_sheet, "ERROR_FOCUS", colors.error_focus);
  replace_token(style_sheet, "ERROR_CHECK", colors.error_check);
  replace_token(style_sheet, "SCROLL", colors.scroll);
  replace_token(style_sheet, "HIGHLIGHTED_TEXT", colors.highlighted_text);
  return style_sheet;
}

} // namespace

void apply_dark_clinical_theme(QApplication &app) {
  const ClinicalDarkThemeColors colors;

  app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
  app.setPalette(build_palette(colors));
  app.setStyleSheet(load_clinical_qss(colors));
}

void apply_light_clinical_theme(QApplication &app) {
  const ClinicalLightThemeColors colors;

  app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
  app.setPalette(build_palette(colors));
  app.setStyleSheet(load_clinical_qss(colors));
}

} // namespace holovibes::ui
