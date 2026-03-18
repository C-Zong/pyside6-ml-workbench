from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QFileDialog, QListWidgetItem, QMainWindow, QMessageBox, QSplitter, QTabWidget, QVBoxLayout, QWidget

from .controller import AppController
from .models import get_model_options, is_incremental_model
from .state import AppState
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
        self.setWindowIcon(QIcon("assets/icon.ico"))
        screen = QGuiApplication.primaryScreen()
        geometry = screen.availableGeometry()
        self.resize(int(geometry.width() * 0.9), int(geometry.height() * 0.9))

        self.controller = AppController(AppState())
        self.dataset_panel = DatasetPanel()
        self.clean_tab = CleanTab()
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
        self.workflow_tabs.addTab(self.merge_tab, "Merge")
        self.workflow_tabs.addTab(self.train_tab, "Train")
        center_layout.addWidget(self.workflow_tabs)
        splitter.addWidget(center)
        splitter.addWidget(self.detail_panel)
        splitter.setSizes([280, 360, 1100])
        root_layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.dataset_panel.load_btn.clicked.connect(self.load_files)
        self.dataset_panel.dataset_list.itemSelectionChanged.connect(self.on_dataset_changed)
        self.workflow_tabs.currentChanged.connect(self.detail_panel.set_current_page)

        self.clean_tab.clean_btn.clicked.connect(self.controller.clean_current_dataset)
        self.clean_tab.column_combo.currentTextChanged.connect(self.on_clean_column_changed)
        self.clean_tab.drop_column_btn.clicked.connect(self.on_drop_column)
        self.clean_tab.zero_column_btn.clicked.connect(self.on_zero_column)
        self.clean_tab.split_column_btn.clicked.connect(self.on_split_column)
        self.detail_panel.export_btn.clicked.connect(self.export_current_dataset)

        self.clean_tab.delete_rows_btn.clicked.connect(self.on_delete_rows)
        self.clean_tab.drop_bad_rows_btn.clicked.connect(self.on_drop_bad_rows)
        self.clean_tab.create_column_btn.clicked.connect(self.on_create_column)
        self.clean_tab.create_range_dataset_btn.clicked.connect(self.on_create_range_dataset)

        self.merge_tab.merge_btn.clicked.connect(self.on_merge)
        self.merge_tab.left_dataset_combo.currentTextChanged.connect(self._refresh_merge_key_options)
        self.merge_tab.right_dataset_combo.currentTextChanged.connect(self._refresh_merge_key_options)

        self.train_tab.target_combo.currentTextChanged.connect(self.controller.update_target)
        self.train_tab.target_combo.currentTextChanged.connect(self.refresh_incremental_ui)
        self.train_tab.feature_list.itemSelectionChanged.connect(self.refresh_incremental_ui)
        self.train_tab.model_combo.currentTextChanged.connect(self.refresh_incremental_ui)
        self.train_tab.train_btn.clicked.connect(self.on_train)
        self.train_tab.model_combo.addItems(get_model_options("classification"))

        self.controller.datasets_changed.connect(self.update_dataset_lists)
        self.controller.columns_changed.connect(self.update_column_controls)
        self.controller.working_df_changed.connect(self.detail_panel.update_working_preview)
        self.controller.clean_profile_changed.connect(self.clean_tab.profile_box.setPlainText)
        self.controller.clean_figure_changed.connect(self.detail_panel.update_clean_figure)
        self.controller.model_options_changed.connect(self.update_model_options)
        self.controller.training_results_changed.connect(self.detail_panel.update_train_results)
        self.controller.load_button_enabled.connect(self.dataset_panel.load_btn.setEnabled)
        self.controller.train_button_enabled.connect(self.train_tab.train_btn.setEnabled)
        self.controller.status_changed.connect(self.statusBar().showMessage)
        self.controller.error_occurred.connect(self.show_error)
        self.controller.trained_model_changed.connect(self.dataset_panel.trained_model_label.setText)
        self.controller.trained_model_changed.connect(lambda _text: self.refresh_incremental_ui())

    def load_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            "",
            "Data Files (*.xlsx *.xls *.xlsm *.csv)",
        )
        if files:
            self.controller.load_files(files)

    def update_dataset_lists(self, dataset_names: list[str]) -> None:
        self._replace_combo_items(self.merge_tab.left_dataset_combo, dataset_names)
        self._replace_combo_items(self.merge_tab.right_dataset_combo, dataset_names)

        current_active = self.controller.state.active_dataset_name
        self.dataset_panel.dataset_list.clear()
        for name in dataset_names:
            self.dataset_panel.dataset_list.addItem(QListWidgetItem(name))

        if current_active:
            matches = self.dataset_panel.dataset_list.findItems(current_active, Qt.MatchExactly)
            if matches:
                self.dataset_panel.dataset_list.setCurrentItem(matches[0])
        self._refresh_merge_key_options()

    def update_column_controls(self, columns: list[str]) -> None:
        self._replace_combo_items(self.clean_tab.column_combo, columns)
        self._replace_combo_items(self.clean_tab.left_source_combo, columns)
        self._replace_combo_items(self.clean_tab.right_source_combo, columns)
        self._replace_combo_items(self.clean_tab.time_column_combo, columns)
        self._replace_combo_items(self.train_tab.target_combo, columns)

        self.clean_tab.bad_row_columns.clear()
        self.train_tab.feature_list.clear()
        for column in columns:
            self.clean_tab.bad_row_columns.addItem(QListWidgetItem(column))
            self.train_tab.feature_list.addItem(QListWidgetItem(column))

        self._refresh_merge_key_options()
        if self.clean_tab.column_combo.currentText():
            self.on_clean_column_changed(self.clean_tab.column_combo.currentText())
        if self.train_tab.target_combo.currentText():
            self.controller.update_target(self.train_tab.target_combo.currentText())
        self.refresh_incremental_ui()

    def update_model_options(self, model_names: list[str]) -> None:
        self._replace_combo_items(self.train_tab.model_combo, model_names)
        self.refresh_incremental_ui()

    def on_dataset_changed(self) -> None:
        selected = self.dataset_panel.dataset_list.selectedItems()
        if selected:
            self.controller.select_dataset(selected[0].text())

    def on_drop_column(self) -> None:
        self.controller.drop_selected_column(self.clean_tab.column_combo.currentText())

    def on_zero_column(self) -> None:
        self.controller.zero_selected_column(self.clean_tab.column_combo.currentText())

    def on_clean_column_changed(self, column_name: str) -> None:
        self.clean_tab.set_selected_column_context(column_name)
        self.controller.update_clean_column(column_name)

    def export_current_dataset(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current dataset",
            "",
            "Data Files (*.xlsx *.xlsm *.xls *.csv)",
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
        )

    def on_train(self) -> None:
        target = self.train_tab.target_combo.currentText()
        features = [item.text() for item in self.train_tab.feature_list.selectedItems() if item.text() != target]
        self.controller.train_model(
            target=target,
            features=features,
            model_name=self.train_tab.model_combo.currentText(),
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

    def _get_dataset_columns(self, dataset_name: str) -> list[str]:
        df = self.controller.state.dataframes.get(dataset_name)
        return [str(column) for column in df.columns] if df is not None else []

    def _replace_combo_items(self, combo, values: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)
        if current in values:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def refresh_incremental_ui(self) -> None:
        model_name = self.train_tab.model_combo.currentText()
        target = self.train_tab.target_combo.currentText()
        features = [item.text() for item in self.train_tab.feature_list.selectedItems() if item.text() != target]
        can_continue = self.controller.can_continue_training(target, features, model_name)
        is_incremental = is_incremental_model(model_name)

        self.train_tab.incremental_checkbox.blockSignals(True)
        self.train_tab.incremental_checkbox.setEnabled(is_incremental and can_continue)
        if not (is_incremental and can_continue):
            self.train_tab.incremental_checkbox.setChecked(False)
        self.train_tab.incremental_checkbox.blockSignals(False)
