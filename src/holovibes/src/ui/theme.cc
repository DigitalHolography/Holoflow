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

struct ClinicalThemeColors {
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

void set_palette_color(QPalette &palette, QPalette::ColorRole role, const QString &color) {
  palette.setColor(role, QColor(color));
}

void set_disabled_palette_color(QPalette &palette, QPalette::ColorRole role, const QString &color) {
  palette.setColor(QPalette::Disabled, role, QColor(color));
}

QPalette build_palette(const ClinicalThemeColors &colors) {
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

QString load_clinical_qss(const ClinicalThemeColors &colors) {
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
  const ClinicalThemeColors colors;

  app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
  app.setPalette(build_palette(colors));
  app.setStyleSheet(load_clinical_qss(colors));
}

} // namespace holovibes::ui
