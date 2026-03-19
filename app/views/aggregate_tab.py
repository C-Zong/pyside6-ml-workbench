from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from .aggregate_rules_editor import AggregateRulesEditor


class AggregateTab(QWidget):
    """Controls for aggregating one dataset into a new grouped dataset."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.dataset_combo = QComboBox()
        self.group_key_combo = QComboBox()
        self.rules_editor = AggregateRulesEditor()
        self.binary_checkbox = QCheckBox("Convert Aggregate Column To Binary")
        self.binary_column_combo = QComboBox()
        self.binary_threshold_spin = QDoubleSpinBox()
        self.binary_threshold_spin.setRange(-1_000_000_000, 1_000_000_000)
        self.binary_threshold_spin.setDecimals(4)
        self.binary_threshold_spin.setValue(0)
        self.binary_output_column = QLineEdit()
        self.binary_output_column.setPlaceholderText("Leave Blank To Overwrite")
        self.output_dataset_name = QLineEdit()
        self.output_dataset_name.setPlaceholderText("New Dataset Name")
        self.create_btn = QPushButton("Create Aggregated Dataset")

        layout.addWidget(QLabel("Dataset"))
        layout.addWidget(self.dataset_combo)
        layout.addWidget(QLabel("Group Key"))
        layout.addWidget(self.group_key_combo)
        layout.addWidget(self.rules_editor)
        layout.addWidget(self.binary_checkbox)
        layout.addWidget(QLabel("Binary Source Column"))
        layout.addWidget(self.binary_column_combo)
        layout.addWidget(QLabel("Binary Threshold"))
        layout.addWidget(self.binary_threshold_spin)
        layout.addWidget(QLabel("Binary Output Column"))
        layout.addWidget(self.binary_output_column)
        layout.addWidget(QLabel("Output Dataset Name"))
        layout.addWidget(self.output_dataset_name)
        layout.addWidget(self.create_btn)
        layout.addStretch()

        self.binary_column_combo.setEnabled(False)
        self.binary_threshold_spin.setEnabled(False)
        self.binary_output_column.setEnabled(False)

    def reset_rules(self) -> None:
        """Clear the aggregate-rule editor and list."""
        self.rules_editor.reset_rules()
        self.binary_column_combo.clear()
        self.binary_checkbox.setChecked(False)
        self.binary_output_column.clear()
        self.output_dataset_name.clear()
