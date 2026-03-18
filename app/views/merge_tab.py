from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget


class MergeTab(QWidget):
    """Controls for joining or appending datasets."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.left_dataset_combo = QComboBox()
        self.right_dataset_combo = QComboBox()
        self.left_key_combo = QComboBox()
        self.right_key_combo = QComboBox()
        self.join_type_combo = QComboBox()
        self.join_type_combo.addItems(["inner", "left", "right", "outer", "append"])
        self.merge_btn = QPushButton("Merge Datasets")

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
        layout.addWidget(self.merge_btn)
        layout.addStretch()
