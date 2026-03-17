from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QTableView,
    QTextEdit,
    QSplitter,
    QTabWidget,
    QSizePolicy,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from .state import AppState
from .table_model import DataFrameTableModel
from .data_ops import load_table, merge_tables, column_profile, clean_dataframe, infer_task_type
from .plotting import build_distribution_figure, build_importance_figure
from .worker import TrainWorker
from .models import get_model_options


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Excel ML Desktop Tool")
        self.setWindowIcon(QIcon("assets/icon.ico"))
        self.resize(1500, 900)

        self.state = AppState()
        self.table_model = DataFrameTableModel()
        self.current_thread = None
        self.current_worker = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("Load Excel/CSV")
        self.merge_btn = QPushButton("Merge Selected Files")
        self.clean_btn = QPushButton("Clean Data")
        self.train_btn = QPushButton("Train Model")

        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.merge_btn)
        toolbar.addWidget(self.clean_btn)
        toolbar.addWidget(self.train_btn)
        toolbar.addStretch()

        root_layout.addLayout(toolbar)

        main_splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("Loaded datasets"))
        self.dataset_list = QListWidget()
        left_layout.addWidget(self.dataset_list)

        left_layout.addWidget(QLabel("Left join key / target column"))
        self.target_combo = QComboBox()
        left_layout.addWidget(self.target_combo)

        left_layout.addWidget(QLabel("Right join key"))
        self.right_key_combo = QComboBox()
        left_layout.addWidget(self.right_key_combo)

        left_layout.addWidget(QLabel("Join type"))
        self.join_type_combo = QComboBox()
        self.join_type_combo.addItems(["inner", "left", "right", "outer"])
        left_layout.addWidget(self.join_type_combo)

        left_layout.addWidget(QLabel("Feature columns"))
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        left_layout.addWidget(self.feature_list)

        left_layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_model_options("classification"))
        left_layout.addWidget(self.model_combo)

        left_layout.addWidget(QLabel("Column profile / metrics"))
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        left_layout.addWidget(self.info_box)

        right_tabs = QTabWidget()

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        preview_layout.addWidget(self.table_view)

        distribution_tab = QWidget()
        self.distribution_layout = QVBoxLayout(distribution_tab)

        result_tab = QWidget()
        self.result_layout = QVBoxLayout(result_tab)

        right_tabs.addTab(preview_tab, "Preview")
        right_tabs.addTab(distribution_tab, "Distribution")
        right_tabs.addTab(result_tab, "Results")

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_tabs)
        main_splitter.setSizes([350, 1100])
        root_layout.addWidget(main_splitter)

        self.load_btn.clicked.connect(self.load_files)
        self.merge_btn.clicked.connect(self.merge_selected)
        self.clean_btn.clicked.connect(self.clean_current_data)
        self.train_btn.clicked.connect(self.train_current_model)
        self.dataset_list.itemSelectionChanged.connect(self.on_dataset_changed)
        self.feature_list.itemSelectionChanged.connect(self.on_feature_selection_changed)
        self.target_combo.currentTextChanged.connect(self.on_target_changed)

    def load_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            "",
            "Data Files (*.xlsx *.xls *.csv)",
        )
        if not files:
            return

        for file_path in files:
            try:
                df = load_table(file_path)
                name = Path(file_path).name
                self.state.dataframes[name] = df
            except Exception as exc:
                QMessageBox.critical(self, "Load error", f"Failed to load {file_path}\n{exc}")

        self.refresh_dataset_list()

    def refresh_dataset_list(self) -> None:
        self.dataset_list.clear()
        for name in self.state.dataframes.keys():
            self.dataset_list.addItem(name)

    def get_active_df(self):
        if self.state.cleaned_df is not None:
            return self.state.cleaned_df
        if self.state.merged_df is not None:
            return self.state.merged_df
        selected = self.dataset_list.selectedItems()
        if selected:
            return self.state.dataframes[selected[0].text()]
        return None

    def on_dataset_changed(self) -> None:
        selected = self.dataset_list.selectedItems()
        if not selected:
            return
        df = self.state.dataframes[selected[0].text()]
        self.table_model.set_dataframe(df.head(500))
        self.populate_column_controls(df)

    def populate_column_controls(self, df) -> None:
        self.target_combo.clear()
        self.right_key_combo.clear()
        self.feature_list.clear()

        columns = [str(c) for c in df.columns]
        self.target_combo.addItems(columns)
        self.right_key_combo.addItems(columns)
        for col in columns:
            item = QListWidgetItem(col)
            self.feature_list.addItem(item)

    def merge_selected(self) -> None:
        items = self.dataset_list.selectedItems()
        if len(items) != 2:
            QMessageBox.warning(self, "Merge", "Please select exactly 2 datasets.")
            return

        left_name = items[0].text()
        right_name = items[1].text()
        left_df = self.state.dataframes[left_name]
        right_df = self.state.dataframes[right_name]
        left_key = self.target_combo.currentText()
        right_key = self.right_key_combo.currentText()
        how = self.join_type_combo.currentText()

        try:
            merged = merge_tables(left_df, right_df, left_key, right_key, how)
            self.state.merged_df = merged
            self.state.cleaned_df = None
            self.table_model.set_dataframe(merged.head(500))
            self.populate_column_controls(merged)
            self.info_box.setPlainText(f"Merged shape: {merged.shape}")
        except Exception as exc:
            QMessageBox.critical(self, "Merge error", str(exc))

    def clean_current_data(self) -> None:
        df = self.get_active_df()
        if df is None:
            QMessageBox.warning(self, "Clean", "No dataset available.")
            return

        try:
            cleaned = clean_dataframe(df)
            self.state.cleaned_df = cleaned
            self.table_model.set_dataframe(cleaned.head(500))
            self.populate_column_controls(cleaned)
            self.info_box.setPlainText(
                f"Cleaned data\nRows: {cleaned.shape[0]}\nCols: {cleaned.shape[1]}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Clean error", str(exc))

    def on_feature_selection_changed(self) -> None:
        df = self.get_active_df()
        selected_items = self.feature_list.selectedItems()
        if df is None or not selected_items:
            return
        col = selected_items[0].text()
        profile = column_profile(df, col)
        lines = [f"{k}: {v}" for k, v in profile.items()]
        self.info_box.setPlainText("\n".join(lines))
        self.render_distribution(df, col)

    def on_target_changed(self) -> None:
        df = self.get_active_df()
        target = self.target_combo.currentText()
        if df is None or not target:
            return
        task_type = infer_task_type(df, target)
        self.state.task_type = task_type
        self.model_combo.clear()
        self.model_combo.addItems(get_model_options(task_type))

    def render_distribution(self, df, col: str) -> None:
        self.clear_layout(self.distribution_layout)
        fig = build_distribution_figure(df, col)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.distribution_layout.addWidget(canvas)

    def train_current_model(self) -> None:
        df = self.get_active_df()
        if df is None:
            QMessageBox.warning(self, "Train", "No active dataset.")
            return

        target = self.target_combo.currentText()
        features = [item.text() for item in self.feature_list.selectedItems() if item.text() != target]
        if not target or not features:
            QMessageBox.warning(self, "Train", "Please choose target and at least one feature.")
            return

        self.state.feature_columns = features
        self.state.target_column = target
        self.state.selected_model_name = self.model_combo.currentText()

        self.train_btn.setEnabled(False)
        self.info_box.setPlainText("Training started...")

        self.current_thread = QThread()
        self.current_worker = TrainWorker(
            df=df,
            features=features,
            target=target,
            model_name=self.state.selected_model_name,
            task_type=self.state.task_type,
        )
        self.current_worker.moveToThread(self.current_thread)
        self.current_thread.started.connect(self.current_worker.run)
        self.current_worker.finished.connect(self.on_train_finished)
        self.current_worker.error.connect(self.on_train_error)
        self.current_worker.finished.connect(self.current_thread.quit)
        self.current_worker.error.connect(self.current_thread.quit)
        self.current_thread.finished.connect(self.current_thread.deleteLater)
        self.current_thread.start()

    def on_train_finished(self, result) -> None:
        self.train_btn.setEnabled(True)
        self.state.trained_model = result.pipeline
        self.state.metrics = result.metrics
        self.state.feature_importance = result.importance_df

        metric_lines = [f"{k}: {v:.4f}" for k, v in result.metrics.items()]
        self.info_box.setPlainText("Training complete\n" + "\n".join(metric_lines))

        self.clear_layout(self.result_layout)
        if not result.importance_df.empty:
            fig = build_importance_figure(result.importance_df)
            canvas = FigureCanvas(fig)
            self.result_layout.addWidget(canvas)

    def on_train_error(self, message: str) -> None:
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Training error", message)
        self.info_box.setPlainText(message)

    @staticmethod
    def clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()