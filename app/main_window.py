from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog, QListWidgetItem, QMainWindow, QMessageBox, QSplitter, QTabWidget, QVBoxLayout, QWidget

from .controller import AppController
from .models import get_model_options, is_incremental_model
from .resources import resource_path
from .state import AppState
from .views.aggregate_tab import AggregateTab
from .views.clean_tab import CleanTab
from .views.dataset_panel import DatasetPanel
from .views.detail_panel import DetailPanel
from .views.merge_tab import MergeTab
from .views.train_tab import TrainTab


class MainWindow(QMainWindow):
    """Compose the workbench UI from shared panels and focused workflow tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Excel ML Desktop Tool")
        self.setWindowIcon(QIcon(resource_path("assets/icon.ico")))

        self.controller = AppController(AppState())
        self.dataset_panel = DatasetPanel()
        self.clean_tab = CleanTab()
        self.aggregate_tab = AggregateTab()
        self.merge_tab = MergeTab()
        self.train_tab = TrainTab()
        self.detail_panel = DetailPanel()

        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage("Load a dataset to start.")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        splitter = QSplitter()
        splitter.addWidget(self.dataset_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.addTab(self.clean_tab, "Clean")
        self.workflow_tabs.addTab(self.aggregate_tab, "Aggregate")
        self.workflow_tabs.addTab(self.merge_tab, "Merge")
        self.workflow_tabs.addTab(self.train_tab, "Train")
        center_layout.addWidget(self.workflow_tabs)
        splitter.addWidget(center)
        splitter.addWidget(self.detail_panel)
        splitter.setSizes([280, 360, 1100])
        root_layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._connect_shared_signals()
        self._connect_clean_signals()
        self._connect_aggregate_signals()
        self._connect_merge_signals()
        self._connect_train_signals()
        self._connect_controller_signals()

    def _connect_shared_signals(self) -> None:
        self.dataset_panel.load_btn.clicked.connect(self.load_files)
        self.dataset_panel.load_model_btn.clicked.connect(self.load_model)
        self.dataset_panel.dataset_list.itemSelectionChanged.connect(self.on_dataset_changed)
        self.dataset_panel.trained_model_list.itemSelectionChanged.connect(self.on_model_changed)
        self.dataset_panel.dataset_list.customContextMenuRequested.connect(self.on_dataset_context_menu)
        self.dataset_panel.trained_model_list.customContextMenuRequested.connect(self.on_model_context_menu)
        self.workflow_tabs.currentChanged.connect(self.detail_panel.set_current_page)

    def _connect_clean_signals(self) -> None:
        self.clean_tab.column_combo.currentTextChanged.connect(self.on_clean_column_changed)
        self.clean_tab.drop_column_btn.clicked.connect(self.on_drop_column)
        self.clean_tab.zero_fill_bad_values_btn.clicked.connect(self.on_zero_fill_bad_values)
        self.clean_tab.split_column_btn.clicked.connect(self.on_split_column)
        self.clean_tab.delete_rows_btn.clicked.connect(self.on_delete_rows)
        self.clean_tab.drop_bad_rows_btn.clicked.connect(self.on_drop_bad_rows)
        self.clean_tab.create_column_btn.clicked.connect(self.on_create_column)
        self.clean_tab.create_range_dataset_btn.clicked.connect(self.on_create_range_dataset)
        self.detail_panel.export_btn.clicked.connect(self.export_current_dataset)

    def _connect_aggregate_signals(self) -> None:
        self.aggregate_tab.dataset_combo.currentTextChanged.connect(self._refresh_aggregate_column_options)
        self._connect_rules_editor(
            self.aggregate_tab.rules_editor,
            self._add_aggregate_rule_from_tab,
            self._clear_aggregate_rules,
        )
        self.aggregate_tab.binary_checkbox.toggled.connect(self._toggle_aggregate_binary_controls)
        self.aggregate_tab.create_btn.clicked.connect(self.on_create_aggregated_dataset)

    def _connect_merge_signals(self) -> None:
        self.merge_tab.merge_btn.clicked.connect(self.on_merge)
        self.merge_tab.left_dataset_combo.currentTextChanged.connect(self._refresh_merge_key_options)
        self.merge_tab.right_dataset_combo.currentTextChanged.connect(self._refresh_merge_key_options)
        self.merge_tab.aggregate_checkbox.toggled.connect(self._toggle_merge_aggregate_tab)
        self.merge_tab.coalesce_checkbox.toggled.connect(self.merge_tab.coalesce_strategy_combo.setEnabled)
        self.merge_tab.left_key_combo.currentTextChanged.connect(self.refresh_merge_risk)
        self.merge_tab.right_key_combo.currentTextChanged.connect(self.refresh_merge_risk)
        self._connect_rules_editor(
            self.merge_tab.rules_editor,
            lambda: self.add_aggregate_rule(self.merge_tab.rules_editor),
            self.merge_tab.rules_editor.rules_list.clear,
        )

    def _connect_train_signals(self) -> None:
        self.train_tab.target_combo.currentTextChanged.connect(self.refresh_training_task_type)
        self.train_tab.task_type_combo.currentTextChanged.connect(self.refresh_training_task_type)
        self.train_tab.feature_list.itemSelectionChanged.connect(self.refresh_incremental_ui)
        self.train_tab.model_combo.currentTextChanged.connect(self.refresh_incremental_ui)
        self.train_tab.model_combo.currentTextChanged.connect(self.train_tab.update_model_params)
        self.train_tab.train_btn.clicked.connect(self.on_train)
        self.train_tab.predict_btn.clicked.connect(self.controller.predict_current_dataset)
        self.train_tab.model_combo.addItems(get_model_options("classification"))

    def _connect_controller_signals(self) -> None:
        self.controller.datasets_changed.connect(self.update_dataset_lists)
        self.controller.columns_changed.connect(self.update_column_controls)
        self.controller.working_df_changed.connect(self.detail_panel.update_working_preview)
        self.controller.clean_profile_changed.connect(self.clean_tab.profile_box.setPlainText)
        self.controller.clean_figure_changed.connect(self.detail_panel.update_clean_figure)
        self.controller.model_options_changed.connect(self.update_model_options)
        self.controller.training_results_changed.connect(self.detail_panel.update_train_results)
        self.controller.load_button_enabled.connect(self.dataset_panel.load_btn.setEnabled)
        self.controller.aggregate_button_enabled.connect(self.aggregate_tab.create_btn.setEnabled)
        self.controller.merge_button_enabled.connect(self.merge_tab.merge_btn.setEnabled)
        self.controller.train_button_enabled.connect(self.train_tab.train_btn.setEnabled)
        self.controller.status_changed.connect(self.statusBar().showMessage)
        self.controller.error_occurred.connect(self.show_error)
        self.controller.trained_models_changed.connect(self.dataset_panel.set_trained_model_items)
        self.controller.trained_model_changed.connect(lambda text: self.train_tab.predict_btn.setEnabled(bool(text.strip())))
        self.controller.trained_model_changed.connect(lambda _text: self.refresh_incremental_ui())
        self.controller.merge_risk_changed.connect(self.merge_tab.merge_risk_box.setPlainText)

    def load_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            "",
            "Data Files (*.xlsx *.xls *.xlsm *.csv)",
        )
        if files:
            self.controller.load_files(files)

    def load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "Model Files (*.pkl)",
        )
        if file_path:
            self.controller.load_trained_model(file_path)

    def update_dataset_lists(self, dataset_names: list[str]) -> None:
        self._sync_aggregate_dataset_controls(dataset_names)
        self._sync_merge_dataset_controls(dataset_names)
        self.aggregate_tab.reset_rules()

        current_active = self.controller.state.active_dataset_name
        self._replace_selectable_list(self.dataset_panel.dataset_list, dataset_names, current_active)
        self._refresh_merge_key_options()

    def update_column_controls(self, columns: list[str]) -> None:
        self._sync_clean_column_controls(columns)
        self._sync_train_column_controls(columns)

        self._refresh_merge_key_options()
        if self.clean_tab.column_combo.currentText():
            self.on_clean_column_changed(self.clean_tab.column_combo.currentText())
        if self.train_tab.target_combo.currentText():
            self.refresh_training_task_type()
        self.refresh_incremental_ui()

    def update_model_options(self, model_names: list[str]) -> None:
        self._replace_combo_items(self.train_tab.model_combo, model_names)
        self.train_tab.update_model_params(self.train_tab.model_combo.currentText())
        self.refresh_incremental_ui()

    def on_dataset_changed(self) -> None:
        selected = self.dataset_panel.dataset_list.selectedItems()
        if selected:
            self.controller.select_dataset(selected[0].text())

    def on_dataset_context_menu(self, position) -> None:
        action, dataset_name = self.dataset_panel.show_dataset_context_menu(position)
        if action == "delete" and dataset_name:
            self.controller.delete_dataset(dataset_name)
        elif action == "export" and dataset_name:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Dataset",
                f"{dataset_name}.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx *.xlsm *.xls)",
            )
            if file_path:
                self.controller.export_dataset(dataset_name, file_path)

    def on_model_context_menu(self, position) -> None:
        action, model_name = self.dataset_panel.show_model_context_menu(position)
        if action == "delete" and model_name:
            self.controller.delete_trained_model(model_name)
        elif action == "export" and model_name:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Trained Model",
                f"{model_name}.pkl",
                "Model Files (*.pkl)",
            )
            if file_path:
                self.controller.export_trained_model(file_path, model_name=model_name)

    def on_model_changed(self) -> None:
        selected = self.dataset_panel.trained_model_list.selectedItems()
        if selected and selected[0].text() != self.controller.state.active_model_name:
            self.controller.select_trained_model(selected[0].text())

    def on_drop_column(self) -> None:
        columns = [item.text() for item in self.clean_tab.drop_columns_list.selectedItems()]
        self.controller.drop_selected_columns(columns)

    def on_zero_fill_bad_values(self) -> None:
        columns = [item.text() for item in self.clean_tab.zero_fill_columns_list.selectedItems()]
        self.controller.zero_fill_bad_values(columns)

    def on_clean_column_changed(self, column_name: str) -> None:
        self.clean_tab.set_selected_column_context(column_name)
        self.controller.update_clean_column(column_name)

    def export_current_dataset(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current dataset",
            "dataset.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xlsm *.xls)",
        )
        if file_path:
            self.controller.export_current_dataset(file_path)

    def on_delete_rows(self) -> None:
        self.controller.delete_rows(self.clean_tab.delete_rows_input.text())

    def on_drop_bad_rows(self) -> None:
        columns = [item.text() for item in self.clean_tab.bad_row_columns.selectedItems()]
        self.controller.delete_bad_rows(columns, self.clean_tab.bad_ratio_spin.value())

    def on_create_column(self) -> None:
        self.controller.create_column(
            new_column=self.clean_tab.new_column_name.text(),
            left_column=self.clean_tab.left_source_combo.currentText(),
            right_column=self.clean_tab.right_source_combo.currentText(),
            separator=self.clean_tab.separator_input.text(),
            left_width=self.clean_tab.left_pad_spin.value(),
            right_width=self.clean_tab.right_pad_spin.value(),
        )

    def on_split_column(self) -> None:
        self.controller.split_column_by_delimiter(
            source_column=self.clean_tab.column_combo.currentText(),
            left_new_column=self.clean_tab.split_left_name.text(),
            right_new_column=self.clean_tab.split_right_name.text(),
            delimiter=self.clean_tab.split_delimiter_input.text(),
            occurrence=self.clean_tab.split_occurrence_spin.value(),
        )

    def on_merge(self) -> None:
        self.controller.merge_datasets(
            left_name=self.merge_tab.left_dataset_combo.currentText(),
            right_name=self.merge_tab.right_dataset_combo.currentText(),
            left_key=self.merge_tab.left_key_combo.currentText(),
            right_key=self.merge_tab.right_key_combo.currentText(),
            join_type=self.merge_tab.join_type_combo.currentText(),
            aggregate_right=self.merge_tab.aggregate_checkbox.isChecked(),
            aggregate_specs=self.merge_tab.rules_editor.parse_rules(),
            coalesce_columns=self.merge_tab.coalesce_checkbox.isChecked(),
            coalesce_strategy=self.merge_tab.coalesce_strategy_combo.currentData(),
        )
        self.merge_tab.rules_editor.reset_rules()
        self.refresh_merge_risk()

    def on_create_aggregated_dataset(self) -> None:
        rules = self._parse_aggregate_rules()
        binary_config = self._build_aggregate_binary_config(rules)
        if binary_config == {}:
            return
        self.controller.create_aggregated_dataset(
            dataset_name=self.aggregate_tab.dataset_combo.currentText(),
            group_key=self.aggregate_tab.group_key_combo.currentText(),
            specs=rules,
            output_name=self.aggregate_tab.output_dataset_name.text(),
            binary_config=binary_config,
        )

    def on_train(self) -> None:
        target = self.train_tab.target_combo.currentText()
        self.controller.train_model(
            target=target,
            features=self._selected_train_features(target),
            model_name=self.train_tab.model_combo.currentText(),
            model_params=self.train_tab.get_model_params(self.train_tab.model_combo.currentText()),
            continue_training=self.train_tab.incremental_checkbox.isChecked(),
        )

    def on_create_range_dataset(self) -> None:
        self.controller.create_range_dataset(
            time_column=self.clean_tab.time_column_combo.currentText(),
            start_text=self.clean_tab.range_start_input.text(),
            end_text=self.clean_tab.range_end_input.text(),
            dataset_name=self.clean_tab.range_dataset_name.text(),
        )

    def _refresh_merge_key_options(self) -> None:
        left_name = self.merge_tab.left_dataset_combo.currentText()
        right_name = self.merge_tab.right_dataset_combo.currentText()
        left_columns = self._get_dataset_columns(left_name)
        right_columns = self._get_dataset_columns(right_name)
        self._replace_combo_items(self.merge_tab.left_key_combo, left_columns)
        self._replace_combo_items(self.merge_tab.right_key_combo, right_columns)
        self._replace_combo_items(self.merge_tab.rules_editor.value_column_combo, right_columns)
        self.refresh_merge_risk()

    def _refresh_aggregate_column_options(self) -> None:
        dataset_name = self.aggregate_tab.dataset_combo.currentText()
        columns = self._get_dataset_columns(dataset_name)
        self._replace_combo_items(self.aggregate_tab.group_key_combo, columns)
        self._replace_combo_items(self.aggregate_tab.rules_editor.value_column_combo, columns)
        self._refresh_aggregate_binary_columns()

    def _get_dataset_columns(self, dataset_name: str) -> list[str]:
        df = self.controller.state.dataframes.get(dataset_name)
        return [str(column) for column in df.columns] if df is not None else []

    def _replace_combo_items(self, combo, values: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        sorted_values = self._sorted_values(values)
        combo.addItems(sorted_values)
        if current in sorted_values:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def refresh_incremental_ui(self) -> None:
        model_name = self.train_tab.model_combo.currentText()
        target = self.train_tab.target_combo.currentText()
        features = self._selected_train_features(target)
        can_continue = self.controller.can_continue_training(target, features, model_name)
        is_incremental = is_incremental_model(model_name)

        self.train_tab.incremental_checkbox.blockSignals(True)
        self.train_tab.incremental_checkbox.setEnabled(is_incremental and can_continue)
        if not (is_incremental and can_continue):
            self.train_tab.incremental_checkbox.setChecked(False)
        self.train_tab.incremental_checkbox.blockSignals(False)

    def refresh_training_task_type(self) -> None:
        self.controller.update_target(
            self.train_tab.target_combo.currentText(),
            task_type_override=self.train_tab.selected_task_type(),
        )
        self.refresh_incremental_ui()

    def _selected_train_features(self, target: str) -> list[str]:
        return [item.text() for item in self.train_tab.feature_list.selectedItems() if item.text() != target]

    def _sync_merge_dataset_controls(self, dataset_names: list[str]) -> None:
        self._replace_combo_items(self.merge_tab.left_dataset_combo, dataset_names)
        self._replace_combo_items(self.merge_tab.right_dataset_combo, dataset_names)

    def _sync_aggregate_dataset_controls(self, dataset_names: list[str]) -> None:
        self._replace_combo_items(self.aggregate_tab.dataset_combo, dataset_names)
        self._refresh_aggregate_column_options()

    def _sync_clean_column_controls(self, columns: list[str]) -> None:
        self._replace_combo_items(self.clean_tab.column_combo, columns)
        self._replace_combo_items(self.clean_tab.left_source_combo, columns)
        self._replace_combo_items(self.clean_tab.right_source_combo, columns)
        self._replace_combo_items(self.clean_tab.time_column_combo, columns)
        self._replace_list_items(self.clean_tab.drop_columns_list, columns)
        self._replace_list_items(self.clean_tab.zero_fill_columns_list, columns)
        self._replace_list_items(self.clean_tab.bad_row_columns, columns)

    def _sync_train_column_controls(self, columns: list[str]) -> None:
        self._replace_combo_items(self.train_tab.target_combo, columns)
        self._replace_list_items(self.train_tab.feature_list, columns)

    def _replace_list_items(self, widget, values: list[str]) -> None:
        selected = {item.text() for item in widget.selectedItems()}
        widget.clear()
        for value in self._sorted_values(values):
            item = QListWidgetItem(value)
            widget.addItem(item)
            if value in selected:
                item.setSelected(True)

    def _replace_selectable_list(self, widget, values: list[str], current_value: str | None = None) -> None:
        widget.blockSignals(True)
        widget.clear()
        for value in self._sorted_values(values):
            item = QListWidgetItem(value)
            widget.addItem(item)
            if current_value and value == current_value:
                widget.setCurrentItem(item)
        widget.blockSignals(False)

    def _sorted_values(self, values: list[str]) -> list[str]:
        return sorted(values, key=lambda value: str(value).casefold())

    def add_aggregate_rule(self, editor) -> None:
        value_column = editor.value_column_combo.currentText()
        output_column = editor.output_column_input.text().strip()
        if not value_column or not output_column:
            self.show_error("Aggregate rule error", "Please choose a Value Column and Output Column.")
            return
        editor.add_current_rule()

    def _connect_rules_editor(self, editor, add_handler, clear_handler) -> None:
        editor.add_rule_btn.clicked.connect(add_handler)
        editor.clear_rules_btn.clicked.connect(clear_handler)
        editor.rules_list.customContextMenuRequested.connect(editor.show_rule_context_menu)

    def _add_aggregate_rule_from_tab(self) -> None:
        self.add_aggregate_rule(self.aggregate_tab.rules_editor)
        self._refresh_aggregate_binary_columns()

    def _clear_aggregate_rules(self) -> None:
        self.aggregate_tab.rules_editor.rules_list.clear()
        self._refresh_aggregate_binary_columns()

    def _parse_aggregate_rules(self) -> list[dict]:
        return self.aggregate_tab.rules_editor.parse_rules()

    def _refresh_aggregate_binary_columns(self) -> None:
        output_columns = [rule.get("output_column", "").strip() for rule in self._parse_aggregate_rules() if rule.get("output_column", "").strip()]
        self._replace_combo_items(self.aggregate_tab.binary_column_combo, output_columns)
        self._toggle_aggregate_binary_controls(self.aggregate_tab.binary_checkbox.isChecked())

    def _toggle_aggregate_binary_controls(self, enabled: bool) -> None:
        has_columns = self.aggregate_tab.binary_column_combo.count() > 0
        active = enabled and has_columns
        self.aggregate_tab.binary_column_combo.setEnabled(active)
        self.aggregate_tab.binary_threshold_spin.setEnabled(active)
        self.aggregate_tab.binary_output_column.setEnabled(active)

    def _build_aggregate_binary_config(self, rules: list[dict]) -> dict | None:
        if not self.aggregate_tab.binary_checkbox.isChecked():
            return None

        source_column = self.aggregate_tab.binary_column_combo.currentText()
        if not source_column:
            self.show_error("Aggregate error", "Please choose a Binary Source Column.")
            return {}

        valid_columns = {rule.get("output_column", "").strip() for rule in rules}
        if source_column not in valid_columns:
            self.show_error("Aggregate error", "Binary Source Column must match an Aggregate Rule output.")
            return {}

        return {
            "source_column": source_column,
            "threshold": self.aggregate_tab.binary_threshold_spin.value(),
            "output_column": self.aggregate_tab.binary_output_column.text(),
        }

    def refresh_merge_risk(self) -> None:
        self.controller.update_merge_risk(
            left_name=self.merge_tab.left_dataset_combo.currentText(),
            right_name=self.merge_tab.right_dataset_combo.currentText(),
            left_key=self.merge_tab.left_key_combo.currentText(),
            right_key=self.merge_tab.right_key_combo.currentText(),
        )

    def _toggle_merge_aggregate_tab(self, enabled: bool) -> None:
        self.merge_tab.merge_sections.setTabEnabled(1, enabled)
        self.merge_tab.rules_editor.setEnabled(enabled)
        if not enabled and self.merge_tab.merge_sections.currentIndex() == 1:
            self.merge_tab.merge_sections.setCurrentIndex(0)
