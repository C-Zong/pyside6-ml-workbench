from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from ..table_model import DataFrameTableModel


class DetailPanel(QWidget):
    """Right-side panel that switches content by active workflow tab."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.stack = QStackedWidget()
        self.clean_page = self._build_clean_page()
        self.merge_page = self._build_merge_page()
        self.train_page = self._build_train_page()

        self.stack.addWidget(self.clean_page)
        self.stack.addWidget(self.merge_page)
        self.stack.addWidget(self.train_page)
        layout.addWidget(self.stack)

    def _build_clean_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        clean_tabs = QTabWidget()
        clean_tabs.addTab(self._build_clean_preview_page(), "Preview")
        clean_tabs.addTab(self._build_clean_distribution_page(), "Distribution")
        clean_tabs.addTab(self._build_clean_rows_page(), "Rows")
        layout.addWidget(clean_tabs)
        return page

    def _build_clean_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        header.addWidget(QLabel("Working Dataset Preview"))
        header.addStretch()
        self.export_btn = QPushButton("Export")
        header.addWidget(self.export_btn)
        layout.addLayout(header)
        self.clean_table = QTableView()
        self.clean_model = DataFrameTableModel()
        self.clean_table.setModel(self.clean_model)
        layout.addWidget(self.clean_table)
        return page

    def _build_clean_distribution_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(QLabel("Value Distribution"))
        self.clean_figure_host = QWidget()
        self.clean_figure_layout = QVBoxLayout(self.clean_figure_host)
        self.clean_figure_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.clean_figure_host)
        layout.addStretch()
        return page

    def _build_clean_rows_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Head Preview"))
        self.process_head_table = QTableView()
        self.process_head_model = DataFrameTableModel()
        self.process_head_table.setModel(self.process_head_model)
        layout.addWidget(self.process_head_table)
        layout.addWidget(QLabel("Tail Preview"))
        self.process_tail_table = QTableView()
        self.process_tail_model = DataFrameTableModel()
        self.process_tail_table.setModel(self.process_tail_model)
        layout.addWidget(self.process_tail_table)
        return page

    def _build_merge_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Merged Dataset Preview"))
        self.merge_table = QTableView()
        self.merge_model = DataFrameTableModel()
        self.merge_table.setModel(self.merge_model)
        layout.addWidget(self.merge_table)
        return page

    def _build_train_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Training Metrics"))
        self.train_metrics = QTextEdit()
        self.train_metrics.setReadOnly(True)
        layout.addWidget(self.train_metrics)

        self.train_figure_container = QWidget()
        train_figure_container_layout = QVBoxLayout(self.train_figure_container)
        train_figure_container_layout.addWidget(QLabel("Feature Importance"))
        self.train_figure_host = QWidget()
        self.train_figure_layout = QVBoxLayout(self.train_figure_host)
        train_figure_container_layout.addWidget(self.train_figure_host)
        layout.addWidget(self.train_figure_container)
        return page

    def set_current_page(self, index: int) -> None:
        """Switch the visible detail page to match the active tab."""
        self.stack.setCurrentIndex(index)

    def update_working_preview(self, df) -> None:
        """Refresh previews that use the current working dataframe."""
        self.clean_model.set_dataframe(df.head(300))
        self.process_head_model.set_dataframe(df.head(10))
        self.process_tail_model.set_dataframe(df.tail(10))
        self.merge_model.set_dataframe(df.head(300))

    def update_clean_figure(self, figure) -> None:
        """Render the clean-tab distribution figure."""
        self._replace_figure(self.clean_figure_layout, figure)

    def update_train_results(self, metrics_text: str, figure) -> None:
        """Render training metrics and feature-importance chart."""
        self.train_metrics.setPlainText(metrics_text)
        self._replace_figure(self.train_figure_layout, figure)

    def _replace_figure(self, layout, figure) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if figure is None:
            return

        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(canvas)
