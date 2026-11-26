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

#include "ui/preferences_dialog.hh"

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QMouseEvent>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QStringList>
#include <QVBoxLayout>
#include <algorithm>

#include "pipeline/manager.hh"

namespace holovibes::ui {

namespace {
constexpr auto kKeepCurrentConfigKey     = "preferences/always_keep_current_config";

void save_graph_preferences(QSettings &settings,
                            const holoflow::core::GraphSpecDumpPreferences &spec,
                            const holoflow::runtime::GraphCompiledDumpPreferences &compiled) {
  settings.beginGroup("preferences/graph_spec");
  settings.setValue("rankdir", static_cast<int>(spec.rankdir));
  settings.setValue("precision", spec.floating_point_precision);
  settings.setValue("node_name", spec.dump_node_name);
  settings.setValue("node_kind", spec.dump_node_kind);
  settings.setValue("node_settings", spec.dump_node_settings);
  settings.setValue("edge_indices", spec.dump_edge_indices);
  settings.endGroup();

  settings.beginGroup("preferences/graph_compiled");
  settings.setValue("rankdir", static_cast<int>(compiled.rankdir));
  settings.setValue("layout", static_cast<int>(compiled.layout));
  settings.setValue("precision", compiled.floating_point_precision);
  settings.setValue("node_name", compiled.dump_node_name);
  settings.setValue("node_kind", compiled.dump_node_kind);
  settings.setValue("node_settings", compiled.dump_node_settings);
  settings.setValue("node_in_out_tids", compiled.dump_node_in_out_tids);
  settings.setValue("edge_indices", compiled.dump_edge_indices);
  settings.setValue("edge_descriptions", compiled.dump_edge_descriptions);
  settings.setValue("section_info", compiled.dump_section_info);
  settings.setValue("section_stream_addr", compiled.dump_section_stream_addr);
  settings.setValue("resource_info", compiled.dump_resource_info);
  settings.endGroup();
}

QSpinBox *create_spin_box(QWidget *parent, int minimum, int maximum, int value) {
  auto *spin_box = new QSpinBox(parent);
  spin_box->setRange(minimum, maximum);
  spin_box->setValue(value);
  return spin_box;
}

QComboBox *create_combo_box(QWidget *parent, const QStringList &items) {
  auto *combo_box = new QComboBox(parent);
  combo_box->addItems(items);
  return combo_box;
}

} // namespace

void restore_graph_preferences(QSettings &settings,
                               holoflow::core::GraphSpecDumpPreferences &spec,
                               holoflow::runtime::GraphCompiledDumpPreferences &compiled) {
  settings.beginGroup("preferences/graph_spec");
  const int spec_rankdir = settings.value("rankdir", static_cast<int>(spec.rankdir)).toInt();
  if (spec_rankdir >= 0 && spec_rankdir <= 1)
    spec.rankdir = static_cast<holoflow::core::GraphSpecDumpPreferences::Rankdir>(spec_rankdir);
  spec.floating_point_precision =
      std::clamp(settings.value("precision", spec.floating_point_precision).toInt(), 0, 17);
  spec.dump_node_name = settings.value("node_name", spec.dump_node_name).toBool();
  spec.dump_node_kind = settings.value("node_kind", spec.dump_node_kind).toBool();
  spec.dump_node_settings = settings.value("node_settings", spec.dump_node_settings).toBool();
  spec.dump_edge_indices = settings.value("edge_indices", spec.dump_edge_indices).toBool();
  settings.endGroup();

  settings.beginGroup("preferences/graph_compiled");
  const int compiled_rankdir = settings.value("rankdir", static_cast<int>(compiled.rankdir)).toInt();
  if (compiled_rankdir >= 0 && compiled_rankdir <= 1)
    compiled.rankdir =
        static_cast<holoflow::runtime::GraphCompiledDumpPreferences::Rankdir>(compiled_rankdir);
  const int layout = settings.value("layout", static_cast<int>(compiled.layout)).toInt();
  if (layout >= 0 && layout <= 3)
    compiled.layout = static_cast<holoflow::runtime::GraphCompiledDumpPreferences::Layout>(layout);
  compiled.floating_point_precision =
      std::clamp(settings.value("precision", compiled.floating_point_precision).toInt(), 0, 17);
  compiled.dump_node_name = settings.value("node_name", compiled.dump_node_name).toBool();
  compiled.dump_node_kind = settings.value("node_kind", compiled.dump_node_kind).toBool();
  compiled.dump_node_settings = settings.value("node_settings", compiled.dump_node_settings).toBool();
  compiled.dump_node_in_out_tids =
      settings.value("node_in_out_tids", compiled.dump_node_in_out_tids).toBool();
  compiled.dump_edge_indices = settings.value("edge_indices", compiled.dump_edge_indices).toBool();
  compiled.dump_edge_descriptions =
      settings.value("edge_descriptions", compiled.dump_edge_descriptions).toBool();
  compiled.dump_section_info = settings.value("section_info", compiled.dump_section_info).toBool();
  compiled.dump_section_stream_addr =
      settings.value("section_stream_addr", compiled.dump_section_stream_addr).toBool();
  compiled.dump_resource_info = settings.value("resource_info", compiled.dump_resource_info).toBool();
  settings.endGroup();
}


namespace {

class CollapsiblePreferencesGroup : public QGroupBox {
public:
  CollapsiblePreferencesGroup(const QString &title, QWidget *parent)
      : QGroupBox(parent), title_(title) {
    update_title();
  }

  void set_content(QWidget *content) {
    content_ = content;
    content_->hide();
  }

protected:
  void mousePressEvent(QMouseEvent *event) override {
    if (event->position().y() <= 30) {
      expanded_ = !expanded_;
      content_->setVisible(expanded_);
      update_title();
      return;
    }
    QGroupBox::mousePressEvent(event);
  }

private:
  void update_title() {
    setTitle((expanded_ ? QStringLiteral("▴ ") : QStringLiteral("▾ ")) + title_);
  }

  QString  title_;
  QWidget *content_  = nullptr;
  bool     expanded_ = false;
};

class PreferencesDialog : public QDialog {
public:
  using GraphSpecDumpPreferences     = holoflow::core::GraphSpecDumpPreferences;
  using GraphCompiledDumpPreferences = holoflow::runtime::GraphCompiledDumpPreferences;

  PreferencesDialog(QWidget *parent, holovibes::pipeline::Manager &manager)
      : QDialog(parent), manager_(manager) {
    setWindowTitle(tr("Preferences"));
    setMinimumWidth(400);
    resize(600, 500);

    auto *dialog_layout = new QVBoxLayout(this);
    auto *scroll_area   = new QScrollArea(this);
    scroll_area->setObjectName("preferencesScrollArea");
    scroll_area->setWidgetResizable(true);
    scroll_area->setFrameShape(QFrame::NoFrame);
    scroll_area->viewport()->setObjectName("preferencesScrollViewport");
    auto *sections        = new QWidget(scroll_area);
    sections->setObjectName("preferencesSections");
    auto *sections_layout = new QVBoxLayout(sections);

    auto *import_group_box        = new QGroupBox(tr("Import"), this);
    auto *import_form             = new QFormLayout();
    keep_current_config_checkbox_ = new QCheckBox(tr("Always keep current config"), this);
    keep_current_config_checkbox_->setChecked(
        QSettings().value(kKeepCurrentConfigKey, false).toBool());
    keep_current_config_checkbox_->setToolTip(
        tr("Keep the current processing settings, including Time Window and Time Stride, when "
           "opening a new .holo file. When off, load settings saved in the file if available."));
    import_form->addRow(keep_current_config_checkbox_);
    import_group_box->setLayout(import_form);
    sections_layout->addWidget(import_group_box);

    // Dump preferences
    auto *graph_spec_dump_group_box =
        setup_graph_spec_dump_preferences(manager_.get_graph_spec_dump_preferences());
    sections_layout->addWidget(graph_spec_dump_group_box);

    auto *graph_compiled_dump_group_box =
        setup_graph_compiled_dump_preferences(manager_.get_graph_compiled_dump_preferences());
    sections_layout->addWidget(graph_compiled_dump_group_box);

    auto *runtime_failure_group_box = new QGroupBox(tr("Failure Diagnostics"), this);
    auto *runtime_failure_form      = new QFormLayout();
    runtime_failure_graph_checkbox_ = new QCheckBox(this);
    runtime_failure_graph_checkbox_->setChecked(manager_.dump_runtime_failure_graphs());
    runtime_failure_graph_checkbox_->setToolTip(
        tr("Save graph_build_failure.dot when a task rejects an input while the pipeline is being "
           "built, compilation_failure.dot for compiler errors, and runtime_failure.dot for "
           "runtime errors and Windows exceptions."));
    runtime_failure_form->addRow(tr("Dump graph on failure"), runtime_failure_graph_checkbox_);
    runtime_failure_group_box->setLayout(runtime_failure_form);
    sections_layout->addWidget(runtime_failure_group_box);
    sections_layout->addStretch();
    scroll_area->setWidget(sections);
    dialog_layout->addWidget(scroll_area);

    // Apply / Close
    auto *button_box = new QDialogButtonBox(QDialogButtonBox::Close, this);
    apply_button_    = button_box->addButton(tr("Apply"), QDialogButtonBox::ActionRole);
    apply_button_->setDisabled(true);
    dialog_layout->addWidget(button_box);

    connect(apply_button_, &QPushButton::clicked, this, [this]() { update_preferences(); });
    connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect_signals();
  }

private:
  void make_collapsible(CollapsiblePreferencesGroup *group_box, QFormLayout *form) {
    auto *content = new QWidget(group_box);
    content->setLayout(form);
    auto *layout = new QVBoxLayout(group_box);
    layout->addWidget(content);
    group_box->set_content(content);
  }

  QGroupBox *
  setup_graph_spec_dump_preferences(const GraphSpecDumpPreferences &graph_spec_dump_preferences) {
    auto *group_box  = new CollapsiblePreferencesGroup(tr("Graph Dump Spec"), this);
    auto *input_form = new QFormLayout();
    input_form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    graph_spec_dump_preferences_widgets_.rankdir_combo_ =
        create_combo_box(this, QStringList{"LR", "TB"});
    if (graph_spec_dump_preferences.rankdir == GraphSpecDumpPreferences::Rankdir::TopToBottom) {
      graph_spec_dump_preferences_widgets_.rankdir_combo_->setCurrentText("TB");
    }

    input_form->addRow(tr("Graph rankdir"), graph_spec_dump_preferences_widgets_.rankdir_combo_);

    graph_spec_dump_preferences_widgets_.floating_point_precision_spin_ =
        create_spin_box(this, 0, 17, graph_spec_dump_preferences.floating_point_precision);
    graph_spec_dump_preferences_widgets_.floating_point_precision_spin_->setToolTip(
        tr("Number of digits after the decimal point in scientific notation."));
    input_form->addRow(tr("Floating-point precision"),
                       graph_spec_dump_preferences_widgets_.floating_point_precision_spin_);

    graph_spec_dump_preferences_widgets_.node_name_checkbox_ = new QCheckBox(this);
    graph_spec_dump_preferences_widgets_.node_name_checkbox_->setChecked(
        graph_spec_dump_preferences.dump_node_name);
    graph_spec_dump_preferences_widgets_.node_name_checkbox_->setToolTip(
        tr("Include node names in the graph dump."));
    input_form->addRow(tr("Node names"), graph_spec_dump_preferences_widgets_.node_name_checkbox_);

    graph_spec_dump_preferences_widgets_.node_kind_checkbox_ = new QCheckBox(this);
    graph_spec_dump_preferences_widgets_.node_kind_checkbox_->setChecked(
        graph_spec_dump_preferences.dump_node_kind);
    graph_spec_dump_preferences_widgets_.node_kind_checkbox_->setToolTip(
        tr("Include node kinds in the graph dump."));
    input_form->addRow(tr("Node kinds"), graph_spec_dump_preferences_widgets_.node_kind_checkbox_);

    graph_spec_dump_preferences_widgets_.node_settings_checkbox_ = new QCheckBox(this);
    graph_spec_dump_preferences_widgets_.node_settings_checkbox_->setChecked(
        graph_spec_dump_preferences.dump_node_settings);
    graph_spec_dump_preferences_widgets_.node_settings_checkbox_->setToolTip(
        tr("Include node settings in the graph dump."));
    input_form->addRow(tr("Node settings"),
                       graph_spec_dump_preferences_widgets_.node_settings_checkbox_);

    graph_spec_dump_preferences_widgets_.edge_indices_checkbox_ = new QCheckBox(this);
    graph_spec_dump_preferences_widgets_.edge_indices_checkbox_->setChecked(
        graph_spec_dump_preferences.dump_edge_indices);
    graph_spec_dump_preferences_widgets_.edge_indices_checkbox_->setToolTip(
        tr("Include edge indices in the graph dump."));
    input_form->addRow(tr("Edge indices"),
                       graph_spec_dump_preferences_widgets_.edge_indices_checkbox_);

    make_collapsible(group_box, input_form);
    return group_box;
  }

  // TODO add compiled graph preferences args
  QGroupBox *setup_graph_compiled_dump_preferences(
      const GraphCompiledDumpPreferences &graph_compiled_dump_preferences) {
    auto *group_box  = new CollapsiblePreferencesGroup(tr("Graph Compiled Spec"), this);
    auto *input_form = new QFormLayout();
    input_form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    graph_compiled_dump_preferences_widgets_.rankdir_combo_ =
        create_combo_box(this, QStringList{"LR", "TB"});
    if (graph_compiled_dump_preferences.rankdir ==
        GraphCompiledDumpPreferences::Rankdir::TopToBottom) {
      graph_compiled_dump_preferences_widgets_.rankdir_combo_->setCurrentText("TB");
    }

    graph_compiled_dump_preferences_widgets_.rankdir_combo_->setToolTip(
        tr("Direction used by the normal compiled-graph layout."));
    input_form->addRow(tr("Rank direction"),
                       graph_compiled_dump_preferences_widgets_.rankdir_combo_);

    graph_compiled_dump_preferences_widgets_.layout_combo_ =
        create_combo_box(this, QStringList{tr("Normal"), tr("Stairs"), tr("Block"), tr("Snake")});
    graph_compiled_dump_preferences_widgets_.layout_combo_->setCurrentIndex(
        static_cast<int>(graph_compiled_dump_preferences.layout));
    graph_compiled_dump_preferences_widgets_.layout_combo_->setToolTip(
        tr("Arrange sections normally, as horizontal stairs, as a left-aligned block, or in "
           "alternating snake rows."));
    graph_compiled_dump_preferences_widgets_.rankdir_combo_->setEnabled(
        graph_compiled_dump_preferences.layout == GraphCompiledDumpPreferences::Layout::Normal);
    input_form->addRow(tr("Section layout"),
                       graph_compiled_dump_preferences_widgets_.layout_combo_);

    graph_compiled_dump_preferences_widgets_.floating_point_precision_spin_ =
        create_spin_box(this, 0, 17, graph_compiled_dump_preferences.floating_point_precision);
    graph_compiled_dump_preferences_widgets_.floating_point_precision_spin_->setToolTip(
        tr("Number of digits after the decimal point in scientific notation."));
    input_form->addRow(tr("Floating-point precision"),
                       graph_compiled_dump_preferences_widgets_.floating_point_precision_spin_);

    graph_compiled_dump_preferences_widgets_.node_name_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.node_name_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_node_name);
    graph_compiled_dump_preferences_widgets_.node_name_checkbox_->setToolTip(
        tr("Include node names in the compiled graph."));
    input_form->addRow(tr("Node names"),
                       graph_compiled_dump_preferences_widgets_.node_name_checkbox_);

    graph_compiled_dump_preferences_widgets_.node_kind_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.node_kind_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_node_kind);
    graph_compiled_dump_preferences_widgets_.node_kind_checkbox_->setToolTip(
        tr("Include node kinds in the compiled graph."));
    input_form->addRow(tr("Node kinds"),
                       graph_compiled_dump_preferences_widgets_.node_kind_checkbox_);

    graph_compiled_dump_preferences_widgets_.node_settings_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.node_settings_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_node_settings);
    graph_compiled_dump_preferences_widgets_.node_settings_checkbox_->setToolTip(
        tr("Include node settings in the compiled graph."));
    input_form->addRow(tr("Node settings"),
                       graph_compiled_dump_preferences_widgets_.node_settings_checkbox_);

    graph_compiled_dump_preferences_widgets_.node_in_out_tids_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.node_in_out_tids_->setChecked(
        graph_compiled_dump_preferences.dump_node_in_out_tids);
    graph_compiled_dump_preferences_widgets_.node_in_out_tids_->setToolTip(
        tr("Include node input and output TIDs in the compiled graph."));
    input_form->addRow(tr("Node I/O TIDs"),
                       graph_compiled_dump_preferences_widgets_.node_in_out_tids_);

    graph_compiled_dump_preferences_widgets_.edge_indices_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.edge_indices_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_edge_indices);
    graph_compiled_dump_preferences_widgets_.edge_indices_checkbox_->setToolTip(
        tr("Include edge indices in the compiled graph."));
    input_form->addRow(tr("Edge indices"),
                       graph_compiled_dump_preferences_widgets_.edge_indices_checkbox_);

    graph_compiled_dump_preferences_widgets_.edge_desc_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.edge_desc_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_edge_descriptions);
    graph_compiled_dump_preferences_widgets_.edge_desc_checkbox_->setToolTip(
        tr("Include edge descriptions in the compiled graph."));
    input_form->addRow(tr("Edge descriptions"),
                       graph_compiled_dump_preferences_widgets_.edge_desc_checkbox_);

    graph_compiled_dump_preferences_widgets_.section_toggle_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.section_toggle_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_section_info);
    graph_compiled_dump_preferences_widgets_.section_toggle_checkbox_->setToolTip(
        tr("Include section information in the compiled graph."));
    input_form->addRow(tr("Section info"),
                       graph_compiled_dump_preferences_widgets_.section_toggle_checkbox_);

    graph_compiled_dump_preferences_widgets_.section_stream_addr_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.section_stream_addr_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_section_stream_addr);
    graph_compiled_dump_preferences_widgets_.section_stream_addr_checkbox_->setToolTip(
        tr("Include section stream addresses in the compiled graph."));
    input_form->addRow(tr("Section stream addresses"),
                       graph_compiled_dump_preferences_widgets_.section_stream_addr_checkbox_);

    graph_compiled_dump_preferences_widgets_.resources_toggle_checkbox_ = new QCheckBox(this);
    graph_compiled_dump_preferences_widgets_.resources_toggle_checkbox_->setChecked(
        graph_compiled_dump_preferences.dump_resource_info);
    graph_compiled_dump_preferences_widgets_.resources_toggle_checkbox_->setToolTip(
        tr("Include resource information in the compiled graph."));
    input_form->addRow(tr("Resource info"),
                       graph_compiled_dump_preferences_widgets_.resources_toggle_checkbox_);

    make_collapsible(group_box, input_form);
    return group_box;
  }

  void update_preferences() {
    // const auto &specs_ = nullptr;
    // holoflow::core::to_dot(specs_);
    apply_button_->setEnabled(false);
    auto graph_spec_dump_preferences = GraphSpecDumpPreferences{
        .rankdir = graph_spec_dump_preferences_widgets_.rankdir_combo_->currentText() == "LR"
                       ? GraphSpecDumpPreferences::Rankdir::LeftToRight
                       : GraphSpecDumpPreferences::Rankdir::TopToBottom,

        .floating_point_precision =
            graph_spec_dump_preferences_widgets_.floating_point_precision_spin_->value(),

        .dump_node_name = graph_spec_dump_preferences_widgets_.node_name_checkbox_->isChecked(),
        .dump_node_kind = graph_spec_dump_preferences_widgets_.node_kind_checkbox_->isChecked(),
        .dump_node_settings =
            graph_spec_dump_preferences_widgets_.node_settings_checkbox_->isChecked(),
        .dump_edge_indices =
            graph_spec_dump_preferences_widgets_.edge_indices_checkbox_->isChecked()};

    auto graph_compiled_dump_preferences = GraphCompiledDumpPreferences{
        .rankdir = graph_compiled_dump_preferences_widgets_.rankdir_combo_->currentText() == "LR"
                       ? GraphCompiledDumpPreferences::Rankdir::LeftToRight
                       : GraphCompiledDumpPreferences::Rankdir::TopToBottom,
        .layout  = static_cast<GraphCompiledDumpPreferences::Layout>(
            graph_compiled_dump_preferences_widgets_.layout_combo_->currentIndex()),

        .floating_point_precision =
            graph_compiled_dump_preferences_widgets_.floating_point_precision_spin_->value(),

        .dump_node_name = graph_compiled_dump_preferences_widgets_.node_name_checkbox_->isChecked(),
        .dump_node_kind = graph_compiled_dump_preferences_widgets_.node_kind_checkbox_->isChecked(),
        .dump_node_settings =
            graph_compiled_dump_preferences_widgets_.node_settings_checkbox_->isChecked(),
        .dump_node_in_out_tids =
            graph_compiled_dump_preferences_widgets_.node_in_out_tids_->isChecked(),
        .dump_edge_indices =
            graph_compiled_dump_preferences_widgets_.edge_indices_checkbox_->isChecked(),
        .dump_edge_descriptions =
            graph_compiled_dump_preferences_widgets_.edge_desc_checkbox_->isChecked(),
        .dump_section_info =
            graph_compiled_dump_preferences_widgets_.section_toggle_checkbox_->isChecked(),
        .dump_section_stream_addr =
            graph_compiled_dump_preferences_widgets_.section_stream_addr_checkbox_->isChecked(),
        .dump_resource_info =
            graph_compiled_dump_preferences_widgets_.resources_toggle_checkbox_->isChecked()};

    manager_.update_graph_spec_dump_preferences(graph_spec_dump_preferences);
    manager_.update_graph_compiled_dump_preferences(graph_compiled_dump_preferences);
    manager_.set_dump_runtime_failure_graphs(runtime_failure_graph_checkbox_->isChecked());
    QSettings settings;
    settings.setValue(kKeepCurrentConfigKey, keep_current_config_checkbox_->isChecked());
    save_graph_preferences(settings, graph_spec_dump_preferences,
                           graph_compiled_dump_preferences);
    settings.sync();
  }

  void connect_signals() {
    connect(graph_spec_dump_preferences_widgets_.rankdir_combo_,
            qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { apply_button_->setEnabled(true); });
    connect(graph_spec_dump_preferences_widgets_.floating_point_precision_spin_,
            qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int) { apply_button_->setEnabled(true); });
    connect(graph_spec_dump_preferences_widgets_.node_name_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_spec_dump_preferences_widgets_.node_kind_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_spec_dump_preferences_widgets_.node_settings_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_spec_dump_preferences_widgets_.edge_indices_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });

    connect(graph_compiled_dump_preferences_widgets_.rankdir_combo_,
            qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.layout_combo_,
            qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
              graph_compiled_dump_preferences_widgets_.rankdir_combo_->setEnabled(
                  index == static_cast<int>(GraphCompiledDumpPreferences::Layout::Normal));
              apply_button_->setEnabled(true);
            });
    connect(graph_compiled_dump_preferences_widgets_.floating_point_precision_spin_,
            qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.node_name_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.node_kind_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.node_settings_checkbox_, &QCheckBox::toggled,
            this, [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.node_in_out_tids_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.edge_indices_checkbox_, &QCheckBox::toggled,
            this, [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.edge_desc_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.section_toggle_checkbox_, &QCheckBox::toggled,
            this, [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.section_stream_addr_checkbox_,
            &QCheckBox::toggled, this, [this](bool) { apply_button_->setEnabled(true); });
    connect(graph_compiled_dump_preferences_widgets_.resources_toggle_checkbox_,
            &QCheckBox::toggled, this, [this](bool) { apply_button_->setEnabled(true); });
    connect(runtime_failure_graph_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
    connect(keep_current_config_checkbox_, &QCheckBox::toggled, this,
            [this](bool) { apply_button_->setEnabled(true); });
  }

  holovibes::pipeline::Manager &manager_;

  QPushButton *apply_button_ = nullptr;

  struct GraphSpecDumpPreferencesWidgets {
    // dump preferences
    // rankdir: LR | TB
    QComboBox *rankdir_combo_                 = nullptr;
    QSpinBox  *floating_point_precision_spin_ = nullptr;
    // Nodes
    QCheckBox *node_name_checkbox_     = nullptr;
    QCheckBox *node_kind_checkbox_     = nullptr;
    QCheckBox *node_settings_checkbox_ = nullptr;
    // Edges
    QCheckBox *edge_indices_checkbox_ = nullptr;
  };
  GraphSpecDumpPreferencesWidgets graph_spec_dump_preferences_widgets_;

  struct GraphCompiledDumpPreferencesWidgets {
    // dump preferences
    // rankdir: LR | TB
    QComboBox *rankdir_combo_                 = nullptr;
    QComboBox *layout_combo_                  = nullptr;
    QSpinBox  *floating_point_precision_spin_ = nullptr;
    QCheckBox *node_name_checkbox_            = nullptr;
    QCheckBox *node_kind_checkbox_            = nullptr;
    QCheckBox *node_settings_checkbox_        = nullptr;
    QCheckBox *node_in_out_tids_              = nullptr;
    // Edges
    QCheckBox *edge_indices_checkbox_ = nullptr;
    QCheckBox *edge_desc_checkbox_    = nullptr;

    // Section
    QCheckBox *section_toggle_checkbox_      = nullptr;
    QCheckBox *section_stream_addr_checkbox_ = nullptr;

    // Resources
    QCheckBox *resources_toggle_checkbox_ = nullptr;
  };
  GraphCompiledDumpPreferencesWidgets graph_compiled_dump_preferences_widgets_;
  QCheckBox                          *runtime_failure_graph_checkbox_ = nullptr;
  QCheckBox                          *keep_current_config_checkbox_   = nullptr;
};


} // namespace

void show_preferences_dialog(QWidget *parent, pipeline::Manager &manager) {
  PreferencesDialog dialog(parent, manager);
  dialog.exec();
}

} // namespace holovibes::ui
