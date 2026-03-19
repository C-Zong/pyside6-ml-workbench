from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from .aggregate_rules_editor import AggregateRulesEditor


class MergeTab(QWidget):
    """Controls for joining or appending prepared datasets."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.left_dataset_combo = QComboBox()
        self.right_dataset_combo = QComboBox()
        self.left_key_combo = QComboBox()
        self.right_key_combo = QComboBox()
        self.join_type_combo = QComboBox()
        self.join_type_combo.addItems(["inner", "left", "right", "outer", "append"])
        self.aggregate_checkbox = QCheckBox("Aggregate Right Before Merge")
        self.coalesce_checkbox = QCheckBox("Coalesce Columns")
        self.coalesce_strategy_combo = QComboBox()
        self.coalesce_strategy_combo.addItem("First Non-Empty", "first_non_empty")
        self.coalesce_strategy_combo.addItem("Prefer Left", "prefer_left")
        self.coalesce_strategy_combo.addItem("Prefer Right", "prefer_right")
        self.rules_editor = AggregateRulesEditor()
        self.merge_risk_box = QTextEdit()
        self.merge_risk_box.setReadOnly(True)
        self.merge_btn = QPushButton("Merge Datasets")
        self.merge_sections = QTabWidget()

        self.merge_sections.addTab(self._build_basic_tab(), "Basic")
        self.merge_sections.addTab(self._build_aggregate_tab(), "Aggregate")
        self.merge_sections.setTabEnabled(1, False)
        self.merge_sections.tabBar().setExpanding(True)
        self.merge_sections.tabBar().setUsesScrollButtons(False)
        layout.addWidget(self.merge_sections)
        layout.addWidget(self.merge_btn)
        layout.addStretch()

        self.rules_editor.setEnabled(False)
        self.coalesce_strategy_combo.setEnabled(False)

    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Left Dataset"))
        layout.addWidget(self.left_dataset_combo)
        layout.addWidget(QLabel("Right Dataset"))
        layout.addWidget(self.right_dataset_combo)
        layout.addWidget(QLabel("Left Key"))
        layout.addWidget(self.left_key_combo)
        layout.addWidget(QLabel("Right Key"))
        layout.addWidget(self.right_key_combo)
        layout.addWidget(QLabel("Join Type"))
        layout.addWidget(self.join_type_combo)
        layout.addWidget(self.aggregate_checkbox)
        layout.addWidget(self.coalesce_checkbox)
        layout.addWidget(self.coalesce_strategy_combo)
        layout.addWidget(QLabel("Merge Risk"))
        layout.addWidget(self.merge_risk_box)
        layout.addStretch()
        return page

    def _build_aggregate_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.rules_editor)
        return page
