from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class CleanTab(QWidget):
    """Organize cleaning actions into compact grouped tabs."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.column_combo = QComboBox()
        self.selected_column_hint = QLabel("Current Selected Column:")
        self.profile_box = QTextEdit()
        self.profile_box.setReadOnly(True)

        self.clean_btn = QPushButton("Run Default Clean")
        self.drop_column_btn = QPushButton("Delete Column")
        self.zero_column_btn = QPushButton("Set Column To 0")

        self.new_column_name = QLineEdit()
        self.new_column_name.setPlaceholderText("New Column")
        self.left_source_combo = QComboBox()
        self.right_source_combo = QComboBox()
        self.separator_input = QLineEdit("-")
        self.left_pad_spin = QSpinBox()
        self.left_pad_spin.setRange(0, 20)
        self.right_pad_spin = QSpinBox()
        self.right_pad_spin.setRange(0, 20)
        self.right_pad_spin.setValue(2)
        self.create_column_btn = QPushButton("Create Formatted Column")

        self.split_left_name = QLineEdit()
        self.split_left_name.setPlaceholderText("Left Part")
        self.split_right_name = QLineEdit()
        self.split_right_name.setPlaceholderText("Right Part")
        self.split_delimiter_input = QLineEdit()
        self.split_occurrence_spin = QSpinBox()
        self.split_occurrence_spin.setRange(1, 20)
        self.split_occurrence_spin.setValue(1)
        self.split_column_btn = QPushButton("Split Column")

        self.delete_rows_input = QLineEdit()
        self.delete_rows_input.setPlaceholderText("0, 3, 5-8")
        self.delete_rows_btn = QPushButton("Delete Rows By Index")
        self.bad_row_columns = QListWidget()
        self.bad_row_columns.setSelectionMode(QAbstractItemView.MultiSelection)
        self.bad_ratio_spin = QDoubleSpinBox()
        self.bad_ratio_spin.setRange(0.0, 1.0)
        self.bad_ratio_spin.setSingleStep(0.05)
        self.bad_ratio_spin.setValue(0.5)
        self.drop_bad_rows_btn = QPushButton("Delete Rows Above Bad-Row Ratio")

        self.time_column_combo = QComboBox()
        self.range_start_input = QLineEdit()
        self.range_start_input.setPlaceholderText("Start")
        self.range_end_input = QLineEdit()
        self.range_end_input.setPlaceholderText("End")
        self.range_dataset_name = QLineEdit()
        self.range_dataset_name.setPlaceholderText("New Dataset Name")
        self.create_range_dataset_btn = QPushButton("Create Range Dataset")

        layout.addWidget(QLabel("Selected Column"))
        layout.addWidget(self.column_combo)
        layout.addWidget(self.selected_column_hint)

        self.primary_sections = QTabWidget()
        self.primary_sections.addTab(self._build_profile_tab(), "Profile")
        self.primary_sections.addTab(self._build_basic_tab(), "Basic")
        self.primary_sections.addTab(self._build_time_tab(), "Time")
        self.primary_sections.addTab(self._build_transform_tab(), "Transform")
        self.primary_sections.tabBar().setExpanding(True)
        self.primary_sections.tabBar().setUsesScrollButtons(False)
        layout.addWidget(self.primary_sections)

    def _build_profile_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Column Profile"))
        layout.addWidget(self.profile_box)
        return page

    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.clean_btn)
        layout.addWidget(self.drop_column_btn)
        layout.addWidget(self.zero_column_btn)
        layout.addWidget(QLabel("Delete Specific Rows"))
        layout.addWidget(self.delete_rows_input)
        layout.addWidget(self.delete_rows_btn)
        layout.addWidget(QLabel("Columns Used For Bad-Row Ratio"))
        layout.addWidget(self.bad_row_columns)
        layout.addWidget(QLabel("Bad-Row Ratio Threshold"))
        layout.addWidget(self.bad_ratio_spin)
        layout.addWidget(self.drop_bad_rows_btn)
        layout.addStretch()
        return page

    def _build_combine_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("New Column Name"))
        layout.addWidget(self.new_column_name)
        layout.addWidget(QLabel("Left Source Column"))
        layout.addWidget(self.left_source_combo)
        layout.addWidget(QLabel("Right Source Column"))
        layout.addWidget(self.right_source_combo)
        layout.addWidget(QLabel("Separator"))
        layout.addWidget(self.separator_input)
        layout.addWidget(QLabel("Left Zero-Pad Width"))
        layout.addWidget(self.left_pad_spin)
        layout.addWidget(QLabel("Right Zero-Pad Width"))
        layout.addWidget(self.right_pad_spin)
        layout.addWidget(self.create_column_btn)
        layout.addStretch()
        return page

    def _build_split_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Left Output Column"))
        layout.addWidget(self.split_left_name)
        layout.addWidget(QLabel("Right Output Column"))
        layout.addWidget(self.split_right_name)
        layout.addWidget(QLabel("Split Delimiter"))
        layout.addWidget(self.split_delimiter_input)
        layout.addWidget(QLabel("Split At Delimiter Occurrence"))
        layout.addWidget(self.split_occurrence_spin)
        layout.addWidget(self.split_column_btn)
        layout.addStretch()
        return page

    def _build_time_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Time Column"))
        layout.addWidget(self.time_column_combo)
        layout.addWidget(QLabel("Range Start"))
        layout.addWidget(self.range_start_input)
        layout.addWidget(QLabel("Range End"))
        layout.addWidget(self.range_end_input)
        layout.addWidget(QLabel("Output Dataset Name"))
        layout.addWidget(self.range_dataset_name)
        layout.addWidget(self.create_range_dataset_btn)
        layout.addStretch()
        return page

    def _build_transform_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        transform_tabs = QTabWidget()
        transform_tabs.addTab(self._build_combine_tab(), "Combine")
        transform_tabs.addTab(self._build_split_tab(), "Split")
        transform_tabs.tabBar().setExpanding(True)
        transform_tabs.tabBar().setUsesScrollButtons(False)
        layout.addWidget(transform_tabs)
        return page

    def set_selected_column_context(self, column_name: str) -> None:
        """Show the active column and keep it as the default left source."""
        self.selected_column_hint.setText(f"Current Selected Column: {column_name}")
        if column_name:
            self.left_source_combo.setCurrentText(column_name)
